"""                                                            # Docstring opener
Sheriff of Nottingham — PettingZoo ParallelEnv.                # Summary
Score-only rewards, full rules, multi-card bribes, redirects,  # Features
threats, public messages.                                      # Features
"""                                                            # Docstring closer

from __future__ import annotations                             # Forward refs in hints
import os                                                      # Path handling
import yaml                                                    # YAML parser
import numpy as np                                             # Numerics
from enum import IntEnum                                       # Int enum base
from collections import Counter                                # Card counting
from pettingzoo import ParallelEnv                             # MARL env base
from gymnasium.spaces import MultiDiscrete, Box                # Space types


# ============================================================  # Phase section
# Game phases                                                   # Section title
# ============================================================  # Section bottom
class Phase(IntEnum):                                          # Phase enum
    MARKET        = 0                                          # Phase 0
    LOAD_BAG      = 1                                          # Phase 1
    DECLARATION   = 2                                          # Phase 2
    BRIBERY       = 3                                          # Phase 3
    BRIBERY_REPLY = 4                                          # Phase 4
    INSPECTION    = 5                                          # Phase 5
    END_ROUND     = 6                                          # Phase 6

PHASES = list(Phase)                                           # Ordered list


# ============================================================  # Env section
# Environment                                                   # Section title
# ============================================================  # Section bottom
class SheriffEnv(ParallelEnv):                                 # Main env class
    metadata = {"name": "sheriff_v5", "render_modes": []}      # Metadata

    # --------------------------------------------------------  # Init section
    # __init__                                                  # Section title
    # --------------------------------------------------------  # Section bottom
    def __init__(self, config_path: str | None = None):        # Optional override
        if config_path is None:                                # Default path
            config_path = os.path.join(                        # Build path
                os.path.dirname(os.path.abspath(__file__)),    # Same dir
                "config.yaml")                                 # Filename
        with open(config_path) as f:                           # Open config
            self.cfg = yaml.safe_load(f)                       # Parse YAML

        g = self.cfg["game"]                                   # Game block
        self.num_players       = int(g["num_players"])         # Player count
        self.max_sheriff_turns = int(g["max_sheriff_turns"])   # Sheriff rotations
        self.starting_gold     = int(g["starting_gold"])       # Starting gold
        self.max_hand          = int(g["max_hand"])            # Hand cap
        self.max_bag           = int(g["max_bag"])             # Bag cap
        self.max_discard       = int(g["max_discard"])         # Discard cap
        self.max_bribe         = int(g["max_bribe"])           # Gold bribe cap
        self.max_bribe_goods   = int(g.get("max_bribe_goods", 8))  # Goods bribe cap
        self.max_steps         = int(g["max_steps"])           # Truncation
        self.use_royal         = bool(g["use_royal"])          # Royal toggle
        self.n_messages        = int(g["n_messages"])          # Message vocab

        self.legal_goods = self.cfg["legal_goods"]             # Legal specs
        self.contraband  = self.cfg["contraband"]              # Contraband specs
        self.royal_goods = self.cfg["royal_goods"] if self.use_royal else {}  # Royal specs
        self.king_queen  = self.cfg["king_queen_bonuses"]      # KQ bonuses

        r = self.cfg["rewards"]                                # Rewards block
        self.score_shaping_weight = float(r.get("score_shaping_weight", 0.0))  # Per-step weight
        self.score_multiplier     = float(r.get("score_multiplier", 1.0))      # Terminal weight
        self.win_bonus            = float(r.get("win_bonus", 0.0))             # Winner bonus

        self.legal_types = list(self.legal_goods.keys())       # Ordered legal types
        self.legal_idx   = {t: i for i, t in enumerate(self.legal_types)}  # Name→index
        self.n_legal     = len(self.legal_types)               # Number of legal types

        self.full_deck = self._build_deck()                    # Master deck

        self.possible_agents = [f"player_{i}" for i in range(self.num_players)]  # Agent IDs
        self.agents = []                                       # Active agents

        self.action_spaces = {                                 # Action space per agent
            a: MultiDiscrete([                                 # 12-dim vector
                self.max_discard + 1,                          # [0]  num_discard
                3,                                             # [1]  draw_source
                2 ** self.max_hand,                            # [2]  bag_mask
                self.n_legal,                                  # [3]  declare_type
                self.max_bribe + 1,                            # [4]  bribe_gold
                2 ** self.max_bribe_goods,                     # [5]  goods_mask
                self.max_bag + 1,                              # [6]  bag_idx
                self.num_players + 1,                          # [7]  redirect_target
                self.max_bribe + 1,                            # [8]  redirect_gold
                2 ** (self.num_players - 1),                   # [9]  threaten_mask
                2 ** (self.num_players - 1),                   # [10] inspect_mask
                self.n_messages,                               # [11] message
            ])                                                 # Close MultiDiscrete
            for a in self.possible_agents                      # Every agent
        }                                                      # Close dict

        obs_dim = 31 + (self.num_players - 1) * 19             # Obs size
        self.observation_spaces = {                            # Obs space per agent
            a: Box(-1.0, 1.0, (obs_dim,), np.float32)          # Float32 box
            for a in self.possible_agents                      # Every agent
        }                                                      # Close dict

    # --------------------------------------------------------  # Deck section
    # Deck construction                                         # Section title
    # --------------------------------------------------------  # Section bottom
    def _build_deck(self):                                     # Build master deck
        deck = []                                              # Accumulator
        for t, i in self.legal_goods.items():                  # Each legal type
            deck += [{"type": t, "legal": True, "royal": False,  # Card dict
                      "value": i["value"], "penalty": i["penalty"]}] * i["count"]  # Repeat
        for t, i in self.contraband.items():                   # Each contraband
            deck += [{"type": t, "legal": False, "royal": False,  # Card dict
                      "value": i["value"], "penalty": i["penalty"]}] * i["count"]  # Repeat
        for t, i in self.royal_goods.items():                  # Each royal
            deck += [{"type": t, "legal": False, "royal": True,  # Card dict
                      "value": i["value"], "penalty": i["penalty"]}] * i["count"]  # Repeat
        return deck                                            # Total 216 with royals

    # --------------------------------------------------------  # Draw section
    # Draw cards                                                # Section title
    # --------------------------------------------------------  # Section bottom
    def _draw(self, n, source=0):                              # Draw n cards
        out = []                                               # Accumulator
        for _ in range(n):                                     # Repeat
            if source == 0:                                    # Main deck
                if not self.deck:                              # Empty?
                    rest = self.discard_a[:-5] + self.discard_b[:-5]  # Keep top 5 each
                    if not rest:                               # Nothing left
                        break                                  # Stop
                    self.deck = rest                           # Rebuild
                    np.random.shuffle(self.deck)               # Shuffle
                    self.discard_a = self.discard_a[-5:]       # Preserve
                    self.discard_b = self.discard_b[-5:]       # Preserve
                out.append(self.deck.pop())                    # Take card
            elif source == 1:                                  # Pile A
                if not self.discard_a:                         # Empty
                    break                                      # Stop
                out.append(self.discard_a.pop())               # Take card
            else:                                              # Pile B
                if not self.discard_b:                         # Empty
                    break                                      # Stop
                out.append(self.discard_b.pop())               # Take card
        return out                                             # Return cards

    # --------------------------------------------------------  # Reset section
    # Reset                                                     # Section title
    # --------------------------------------------------------  # Section bottom
    def reset(self, seed=None, options=None):                  # Standard reset
        if seed is not None:                                   # Seed provided
            np.random.seed(seed)                               # Seed RNG
        self.agents = self.possible_agents.copy()              # Activate all

        self.deck = self.full_deck.copy()                      # Copy deck
        np.random.shuffle(self.deck)                           # Shuffle
        self.discard_a = []                                    # Empty pile A
        self.discard_b = []                                    # Empty pile B

        self.hands = {a: self._draw(self.max_hand, 0) for a in self.agents}  # Deal
        self.discard_a = self._draw(5, 0)                      # Seed pile A
        self.discard_b = self._draw(5, 0)                      # Seed pile B

        self.gold  = {a: self.starting_gold for a in self.agents}  # Gold
        self.stand = {a: [] for a in self.agents}              # Stands
        self.bags  = {a: [] for a in self.agents}              # Bags

        self.declarations = {}                                 # Declarations
        self.bribes = {a: self._empty_bribe() for a in self.agents}  # Bribes
        self.messages = {a: 0 for a in self.agents}            # Messages
        self.threatened: set[str] = set()                      # Threats

        self.sheriff = str(np.random.choice(self.agents))      # Random sheriff
        self.sheriff_turns = {a: 0 for a in self.agents}       # Turn counts
        self.phase = Phase.MARKET                              # Start phase
        self.step_count = 0                                    # Step counter
        self.done = False                                      # Not done
        self.final_scores = {}                                 # Final scores
        self.tiebreakers = {}                                  # Tiebreakers

        self.prev_totals = {a: self._current_total(a) for a in self.agents}  # Baseline

        return self._observations(), {a: {} for a in self.agents}  # Return

    @staticmethod                                              # No self needed
    def _empty_bribe():                                        # Default bribe record
        return {"gold": 0, "goods_mask": 0, "bag_idx": None,   # Gold, goods, bag
                "redirect_target": None, "redirect_gold": 0}   # Redirect fields

    # --------------------------------------------------------  # Score section
    # Live total score                                          # Section title
    # --------------------------------------------------------  # Section bottom
    def _current_total(self, agent):                           # Compute live score
        gold = self.gold[agent]                                # Current gold
        goods_val = sum(c["value"] for c in self.stand[agent]) # Stand value

        bonus_total = 0                                        # KQ accumulator
        for t, bonus in self.king_queen.items():               # Each legal type
            mine = sum(1 for c in self.stand[agent]            # My count
                       if c["legal"] and c["type"] == t)       # Legal + matching
            others = {                                         # Others' counts
                o: sum(1 for c in self.stand[o]                # Count each
                       if c["legal"] and c["type"] == t)       # Legal + matching
                for o in self.agents if o != agent             # Non-me agents
            }                                                  # End dict
            top_other = max(others.values()) if others else 0  # Highest opponent

            if mine > top_other:                               # Unique king
                bonus_total += bonus["king"]                   # King bonus
            elif mine == top_other and top_other > 0:          # King tie
                tied = sum(1 for v in others.values() if v == mine) + 1  # Tied count
                bonus_total += (bonus["king"] + bonus["queen"]) // tied  # Split
            else:                                              # Not king
                strictly_above = sum(1 for v in others.values() if v > mine)  # Above count
                if strictly_above == 1 and mine > 0:           # Exactly one above
                    queen_ties = sum(1 for v in others.values() if v == mine)  # Tied with me
                    bonus_total += bonus["queen"] // (queen_ties + 1)  # Split queen
        return gold + goods_val + bonus_total                  # Total live score

    # --------------------------------------------------------  # Obs section
    # Observations                                              # Section title
    # --------------------------------------------------------  # Section bottom
    def _obs_for(self, agent):                                 # Obs for one agent
        f = []                                                 # Accumulator
        f.append(self.gold[agent] / 100.0)                     # Gold
        f.append(len(self.hands[agent]) / self.max_hand)       # Hand size
        f.append(sum(c["value"] for c in self.stand[agent]) / 100.0)  # Stand value
        f.append(1.0 if agent == self.sheriff else 0.0)        # Is sheriff
        f.append(len(self.bags[agent]) / self.max_bag)         # Bag size

        own_decl = [0.0] * self.n_legal                        # Declaration one-hot
        if agent in self.declarations:                         # If declared
            own_decl[self.legal_idx[self.declarations[agent][1]]] = 1.0  # Set bit
        f.extend(own_decl)                                     # Append
        f.append(self.declarations.get(agent, (0, self.legal_types[0]))[0] / self.max_bag)  # Count

        f.extend([1.0 if self.phase == p else 0.0 for p in PHASES])  # Phase one-hot

        total = self.max_sheriff_turns * self.num_players      # Total rounds
        f.append(min(sum(self.sheriff_turns.values()) / total, 1.0))  # Progress

        own_msg = [0.0] * self.n_messages                      # Message one-hot
        own_msg[self.messages[agent]] = 1.0                    # Set bit
        f.extend(own_msg)                                      # Append

        f.append(1.0 if agent in self.threatened else 0.0)     # Am I threatened

        b = self.bribes[agent]                                 # Own bribe
        f.append(b["gold"] / max(self.max_bribe, 1))           # Gold offer
        f.append(bin(b["goods_mask"]).count("1") / max(self.max_bribe_goods, 1))  # Goods count
        f.append(1.0 if b["bag_idx"] is not None else 0.0)     # Bag reveal
        f.append(1.0 if b["redirect_target"] is not None else 0.0)  # Redirect

        for o in self.agents:                                  # Each opponent
            if o == agent:                                     # Skip self
                continue                                       # ...
            f.append(len(self.bags[o]) / self.max_bag)         # Bag size
            opp_decl = [0.0] * self.n_legal                    # Decl one-hot
            if o in self.declarations:                         # If declared
                opp_decl[self.legal_idx[self.declarations[o][1]]] = 1.0  # Set bit
            f.extend(opp_decl)                                 # Append
            f.append(self.declarations.get(o, (0, self.legal_types[0]))[0] / self.max_bag)  # Count
            f.append(1.0 if o == self.sheriff else 0.0)        # Is sheriff
            opp_msg = [0.0] * self.n_messages                  # Message one-hot
            opp_msg[self.messages[o]] = 1.0                    # Set bit
            f.extend(opp_msg)                                  # Append
            ob = self.bribes[o]                                # Opp bribe
            f.append(ob["gold"] / max(self.max_bribe, 1))      # Gold
            f.append(bin(ob["goods_mask"]).count("1") / max(self.max_bribe_goods, 1))  # Goods
            f.append(1.0 if ob["redirect_target"] == agent else 0.0)  # Targets me?
            f.append(1.0 if o in self.threatened else 0.0)     # Threatened

        return np.asarray(f, np.float32)                       # Return array

    def _observations(self):                                   # All observations
        return {a: self._obs_for(a) for a in self.agents}      # Per agent

    # --------------------------------------------------------  # Step section
    # Step                                                      # Section title
    # --------------------------------------------------------  # Section bottom
    def step(self, actions):                                   # Advance one phase
        self.step_count += 1                                   # Increment
        rewards = {a: 0.0 for a in self.agents}                # Init rewards

        if   self.phase == Phase.MARKET:        self._market(actions, rewards)        # Phase 0
        elif self.phase == Phase.LOAD_BAG:      self._load_bag(actions, rewards)      # Phase 1
        elif self.phase == Phase.DECLARATION:   self._declare(actions, rewards)       # Phase 2
        elif self.phase == Phase.BRIBERY:       self._bribery(actions, rewards)       # Phase 3
        elif self.phase == Phase.BRIBERY_REPLY: self._bribery_reply(actions, rewards) # Phase 4
        elif self.phase == Phase.INSPECTION:    self._inspect(actions, rewards)       # Phase 5
        elif self.phase == Phase.END_ROUND:     self._end_round(rewards)              # Phase 6

        for a in self.agents:                                  # For each agent
            cur = self._current_total(a)                       # Current score
            delta = cur - self.prev_totals[a]                  # Delta
            rewards[a] += delta * self.score_shaping_weight    # Weighted delta
            self.prev_totals[a] = cur                          # Update cache

        self._advance_phase()                                  # Next phase

        term  = {a: self.done for a in self.agents}            # Terminations
        trunc = {a: self.step_count >= self.max_steps for a in self.agents}  # Truncations

        if self.done:                                          # Game over
            self._award_terminal(rewards)                      # Terminal reward
            self.agents = []                                   # Clear roster

        return self._observations(), rewards, term, trunc, {a: {} for a in self.agents}  # Return

    # --------------------------------------------------------  # Market
    # MARKET                                                    # Section title
    # --------------------------------------------------------  # Section bottom
    def _market(self, actions, rewards):                       # Market phase
        for a in self.agents:                                  # Each agent
            if a == self.sheriff:                              # Sheriff skips
                continue                                       # ...
            act = actions.get(a)                               # Get action
            if act is None:                                    # No action
                continue                                       # Skip
            d = int(min(act[0], self.max_discard, len(self.hands[a])))  # Discard count
            if d:                                              # If discarding
                self.discard_a.extend(self.hands[a][:d])       # To pile A
                self.hands[a] = self.hands[a][d:]              # Remove
            refill = self.max_hand - len(self.hands[a])        # Cards needed
            if refill > 0:                                     # If under cap
                source = int(act[1]) % 3                       # Draw source
                self.hands[a] += self._draw(refill, source)    # Refill

    # --------------------------------------------------------  # Load
    # LOAD_BAG                                                  # Section title
    # --------------------------------------------------------  # Section bottom
    def _load_bag(self, actions, rewards):                     # Load bag phase
        for a in self.agents:                                  # Each agent
            if a == self.sheriff:                              # Sheriff skips
                continue                                       # ...
            act = actions.get(a)                               # Get action
            if act is None:                                    # No action
                continue                                       # Skip
            mask   = int(act[2])                               # Bag mask
            chosen = [c for i, c in enumerate(self.hands[a]) if (mask >> i) & 1][:self.max_bag]  # Selected
            if not chosen and self.hands[a]:                   # At least one
                chosen = [self.hands[a][0]]                    # Force one
            self.bags[a] = chosen                              # Set bag
            rem = list(self.hands[a])                          # Copy hand
            for c in chosen:                                   # For each chosen
                if c in rem:                                   # If present
                    rem.remove(c)                              # Remove
            self.hands[a] = rem                                # Update hand

    # --------------------------------------------------------  # Declare
    # DECLARATION                                               # Section title
    # --------------------------------------------------------  # Section bottom
    def _declare(self, actions, rewards):                      # Declare phase
        for a in self.agents:                                  # Each agent
            if a == self.sheriff:                              # Sheriff skips
                continue                                       # ...
            act = actions.get(a)                               # Get action
            if act is None:                                    # No action
                continue                                       # Skip
            dtype = self.legal_types[int(act[3]) % self.n_legal]  # Declared type
            bag   = self.bags[a]                               # Bag
            self.declarations[a] = (len(bag), dtype)           # Record

    # --------------------------------------------------------  # Bribery
    # BRIBERY                                                   # Section title
    # --------------------------------------------------------  # Section bottom
    def _bribery(self, actions, rewards):                      # Bribery phase
        merchants = [a for a in self.agents if a != self.sheriff]  # Non-sheriffs

        s_act = actions.get(self.sheriff)                      # Sheriff action
        if s_act is not None:                                  # If provided
            mask = int(s_act[9])                               # Threaten mask
            self.threatened = {merchants[i] for i in range(len(merchants)) if (mask >> i) & 1}  # Decode
        else:                                                  # No action
            self.threatened = set()                            # Empty

        for a in self.agents:                                  # Each agent
            act = actions.get(a)                               # Get action
            if act is None:                                    # No action
                continue                                       # Skip
            self.messages[a] = int(act[11]) % self.n_messages  # Message
            if a == self.sheriff:                              # Sheriff skips
                continue                                       # ...

            goods_mask = int(act[5])                           # Goods bitmask
            bag_idx = int(act[6])                              # Bag reveal index
            if bag_idx >= self.max_bag or bag_idx >= len(self.bags[a]):  # Invalid
                bag_idx = None                                 # Treat as none

            ridx = int(act[7])                                 # Redirect index
            if ridx >= len(merchants):                         # Invalid
                rtarget = None                                 # No redirect
            else:                                              # Valid
                cand = merchants[ridx]                         # Candidate
                rtarget = None if cand == a else cand          # Can't self-redirect
            rgold = int(min(act[8], self.max_bribe, self.gold[a]))  # Clamp gold

            self.bribes[a] = {                                 # Store bribe
                "gold": 0,                                     # Set in reply
                "goods_mask": goods_mask,                      # Goods bitmask
                "bag_idx": bag_idx,                            # Bag reveal
                "redirect_target": rtarget,                    # Redirect target
                "redirect_gold": rgold,                        # Redirect gold
            }                                                  # End dict

    # --------------------------------------------------------  # Reply
    # BRIBERY_REPLY                                             # Section title
    # --------------------------------------------------------  # Section bottom
    def _bribery_reply(self, actions, rewards):                # Reply phase
        for a in self.agents:                                  # Each agent
            act = actions.get(a)                               # Get action
            if act is None:                                    # No action
                continue                                       # Skip
            self.messages[a] = int(act[11]) % self.n_messages  # Message
            if a == self.sheriff:                              # Sheriff skips
                continue                                       # ...
            new_gold = int(min(act[4], self.max_bribe, self.gold[a]))  # Clamp
            self.bribes[a]["gold"] = new_gold                  # Set gold

    # --------------------------------------------------------  # Inspect
    # INSPECTION                                                # Section title
    # --------------------------------------------------------  # Section bottom
    def _inspect(self, actions, rewards):                      # Inspection phase
        sheriff   = self.sheriff                               # Current sheriff
        merchants = [a for a in self.agents if a != sheriff]   # Non-sheriffs
        s_act     = actions.get(sheriff)                       # Sheriff action
        mask      = int(s_act[10]) if s_act is not None else 0 # Inspect mask
        inspected = {merchants[i] for i in range(len(merchants)) if (mask >> i) & 1}  # Decode

        for m in merchants:                                    # Each merchant
            if m in inspected:                                 # If inspected
                self._resolve_inspect(m, sheriff)              # Check
            else:                                              # If accepted
                self._resolve_accept(m, sheriff)               # Take bribe

        for m in merchants:                                    # Redirect deals
            rt = self.bribes[m]["redirect_target"]             # Target
            if rt is not None and rt in inspected:             # If target inspected
                self._pay(m, sheriff, self.bribes[m]["redirect_gold"])  # Pay

    def _resolve_inspect(self, merchant, sheriff):             # Resolve bag check
        bag = self.bags[merchant]                              # Bag
        count, dtype = self.declarations.get(merchant, (0, self.legal_types[0]))  # Declaration
        honest = len(bag) == count and all(c["legal"] and c["type"] == dtype for c in bag)  # Truthful

        if honest:                                             # Honest
            fine = sum(c["penalty"] for c in bag)              # Sheriff pays
            self._pay(sheriff, merchant, fine)                 # Transfer
            for c in bag:                                      # Goods to stand
                self.stand[merchant].append(c)                 # Append
        else:                                                  # Lying
            fine = sum(c["penalty"] for c in bag if not c["legal"])  # Illegal penalty
            self._pay(merchant, sheriff, fine)                 # Merchant pays
            for c in bag:                                      # Each card
                if c["legal"] and c["type"] == dtype:          # Honest goods
                    self.stand[merchant].append(c)             # To stand
                else:                                          # Illegal/misdeclared
                    self.discard_a.append(c)                   # Confiscated
        self.bags[merchant] = []                               # Clear bag

    def _resolve_accept(self, merchant, sheriff):              # Resolve accepted bag
        b = self.bribes[merchant]                              # Bribe

        mask = b["goods_mask"]                                 # Goods mask
        if mask:                                               # If cards offered
            snapshot = list(self.stand[merchant][:self.max_bribe_goods])  # Top N
            for i, card in enumerate(snapshot):                # Each card
                if (mask >> i) & 1 and card in self.stand[merchant]:  # If selected
                    self.stand[merchant].remove(card)          # Remove
                    self.stand[sheriff].append(card)           # To sheriff

        if b["bag_idx"] is not None and b["bag_idx"] < len(self.bags[merchant]):  # Bag reveal
            card = self.bags[merchant].pop(b["bag_idx"])       # Extract
            self.stand[sheriff].append(card)                   # To sheriff

        if b["gold"] > 0:                                      # Gold bribe
            self._pay(merchant, sheriff, b["gold"])            # Transfer

        for c in self.bags[merchant]:                          # Remaining bag
            self.stand[merchant].append(c)                     # To stand
        self.bags[merchant] = []                               # Clear bag

    # --------------------------------------------------------  # End round
    # END_ROUND                                                 # Section title
    # --------------------------------------------------------  # Section bottom
    def _end_round(self, rewards):                             # End round
        self.sheriff_turns[self.sheriff] += 1                  # Mark served

        idx = self.agents.index(self.sheriff)                  # Find sheriff
        self.sheriff = self.agents[(idx + 1) % self.num_players]  # Rotate

        for a in self.agents:                                  # Clear state
            self.bags[a] = []                                  # Empty bag
            self.bribes[a] = self._empty_bribe()               # Reset bribe
        self.declarations = {}                                 # Clear declarations
        self.messages = {a: 0 for a in self.agents}            # Reset messages
        self.threatened = set()                                # Clear threats

        for a in self.agents:                                  # Each agent
            refill = self.max_hand - len(self.hands[a])        # Cards needed
            if refill > 0:                                     # If under cap
                self.hands[a] += self._draw(refill, 0)         # Refill

        if all(self.sheriff_turns[a] >= self.max_sheriff_turns for a in self.agents):  # Done?
            self.done = True                                   # End game
            self.final_scores = self._compute_scores()         # Compute

    def _advance_phase(self):                                  # Next phase
        idx = PHASES.index(self.phase)                         # Current index
        self.phase = PHASES[(idx + 1) % len(PHASES)]           # Wrap

    # --------------------------------------------------------  # Pay
    # Payment                                                   # Section title
    # --------------------------------------------------------  # Section bottom
    def _pay(self, payer, payee, amount):                      # Transfer amount
        if amount <= 0:                                        # No-op
            return                                             # Exit
        paid = min(self.gold[payer], amount)                   # Gold portion
        self.gold[payer] -= paid                               # Deduct
        self.gold[payee] += paid                               # Add
        rem = amount - paid                                    # Remaining
        while rem > 0:                                         # Legal goods
            legal = [c for c in self.stand[payer] if c["legal"]]  # Legal cards
            if not legal:                                      # None left
                break                                          # Stop
            card = legal[0]                                    # Take
            self.stand[payer].remove(card)                     # Remove
            self.stand[payee].append(card)                     # Add
            rem -= card["value"]                               # Reduce
        while rem > 0 and self.stand[payer]:                   # Contraband
            card = self.stand[payer][0]                        # Take
            self.stand[payer].remove(card)                     # Remove
            self.stand[payee].append(card)                     # Add
            rem -= card["value"]                               # Reduce

    # --------------------------------------------------------  # Score
    # Final scoring                                             # Section title
    # --------------------------------------------------------  # Section bottom
    def _compute_scores(self):                                 # Final scores
        scores, legal, contra = {}, {}, {}                     # Accumulators
        for a in self.agents:                                  # Each player
            scores[a] = self.gold[a] + sum(c["value"] for c in self.stand[a])  # Gold + goods
            legal[a]  = Counter(c["type"] for c in self.stand[a] if c["legal"])  # Legal counts
            contra[a] = sum(1 for c in self.stand[a] if not c["legal"])  # Contraband count

        for t, bonus in self.king_queen.items():               # Each legal type
            counts = {a: legal[a].get(t, 0) for a in self.agents}  # Count per player
            if max(counts.values()) == 0:                      # Nobody has any
                continue                                       # Skip
            ordered = sorted(counts.items(), key=lambda kv: -kv[1])  # Descending
            top = ordered[0][1]                                # Top count
            kings = [a for a, c in ordered if c == top]        # Top players
            if len(kings) == 1:                                # Unique king
                scores[kings[0]] += bonus["king"]              # King bonus
                rest = [(a, c) for a, c in ordered if c < top and c > 0]  # Rest
                if rest:                                       # If anyone
                    q_count = rest[0][1]                       # Next highest
                    queens = [a for a, c in rest if c == q_count]  # Queen candidates
                    share = bonus["queen"] // len(queens)      # Split
                    for q in queens:                           # Each queen
                        scores[q] += share                     # Add
            else:                                              # King tie
                combined = bonus["king"] + bonus["queen"]      # Combined
                share = combined // len(kings)                 # Split
                for k in kings:                                # Each tied king
                    scores[k] += share                         # Add

        self.tiebreakers = {a: {"legal": sum(legal[a].values()),  # Legal count
                                "contraband": contra[a]} for a in self.agents}  # Contraband count
        return scores                                          # Return

    # --------------------------------------------------------  # Terminal
    # Terminal reward                                           # Section title
    # --------------------------------------------------------  # Section bottom
    def _award_terminal(self, rewards):                        # Final rewards
        if not self.final_scores:                              # If not computed
            self.final_scores = self._compute_scores()         # Compute

        def rank_key(a):                                       # Ranking key
            tb = self.tiebreakers.get(a, {"legal": 0, "contraband": 0})  # Tiebreak
            return (-self.final_scores[a], -tb["legal"], -tb["contraband"])  # Sort

        ranked = sorted(self.agents, key=rank_key)             # Sorted agents
        winner = ranked[0]                                     # Winner

        for a in self.agents:                                  # Each agent
            rewards[a] += self.final_scores[a] * self.score_multiplier  # Score
            if a == winner:                                    # If winner
                rewards[a] += self.win_bonus                   # Add win bonus


# ============================================================  # Demo
if __name__ == "__main__":                                     # Run directly
    env = SheriffEnv()                                         # Instantiate
    print(f"Agents: {env.possible_agents}")                    # Print agents
    print(f"Action: {env.action_spaces['player_0']}")          # Action space
    print(f"Obs:    {env.observation_spaces['player_0']}")     # Obs space
    print(f"Deck:   {len(env.full_deck)}")                     # Deck size
    print(f"Bonus:  win={env.win_bonus}, shaping={env.score_shaping_weight}")  # Weights

    obs, _ = env.reset(seed=42)                                # Reset
    totals = {a: 0.0 for a in env.agents}                      # Totals
    while env.agents:                                          # While alive
        acts = {a: env.action_spaces[a].sample() for a in env.agents}  # Random
        obs, rew, term, trunc, _ = env.step(acts)              # Step
        for a, r in rew.items():                               # Accumulate
            totals[a] += r                                     # Add
        if any(trunc.values()):                                # If truncated
            break                                              # Stop

    print(f"Steps:  {env.step_count}")                         # Steps
    print(f"Scores: {env.final_scores}")                       # Scores
    print(f"Rewards: { {a: round(v, 2) for a, v in totals.items()} }")  # Rewards
    print("✅ Env works.")                                     # Done
