import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
import os

@dataclass
class UserState:
    user_id: str
    z_long: np.ndarray   # [k]
    z_short: np.ndarray  # [k]
    reward_ma: float     # baseline for reward, init 0.0

class UserTensorStore:
    def __init__(self, k: int, path: str):
        self.k = k
        self.path = path
        self._states: Dict[str, UserState] = {}
        self._load()
        
        # Calculate global mean for initialization
        if self._states:
            z_all = np.stack([st.z_long for st in self._states.values()])
            self.global_init_z = np.mean(z_all, axis=0)
        else:
            self.global_init_z = np.zeros(self.k, dtype=np.float32)

    def _load(self):
        if os.path.exists(self.path):
            try:
                data = np.load(self.path, allow_pickle=True)
                # Assume saved as dict of user_id -> dict/object
                # For simplicity, let's say we save a single dict in a .npy or .npz
                # But np.save/load with pickle is tricky for complex objects.
                # Let's save as .npz where each key is user_id and value is a structured array or just use z_long for now?
                # A robust way for prototype:
                # save multiple arrays: "u1_long", "u1_short", "u1_meta"
                pass 
                # For Day 2 prototype, we might just re-init from init script or rely on memory if not persisting strictly.
                # But let's try to load if we can.
                
                # Let's implement a simple npz schema:
                # keys: "{uid}_long", "{uid}_short", "{uid}_meta" (meta=[reward_ma])
                for key in data.files:
                    if key.endswith("_long"):
                        uid = key[:-5]
                        z_long = data[key]
                        z_short = data.get(f"{uid}_short", np.zeros(self.k))
                        meta = data.get(f"{uid}_meta", np.array([0.0]))
                        self._states[uid] = UserState(uid, z_long, z_short, float(meta[0]))
            except Exception as e:
                print(f"Warning: Failed to load UserStore from {self.path}: {e}")

    def _save(self):
        # Save to npz
        save_dict = {}
        for uid, state in self._states.items():
            save_dict[f"{uid}_long"] = state.z_long
            save_dict[f"{uid}_short"] = state.z_short
            save_dict[f"{uid}_meta"] = np.array([state.reward_ma])
        np.savez(self.path, **save_dict)

    def get_state(self, user_id: str) -> UserState:
        if user_id not in self._states:
            # Lazy init with global mean for new users
            state = UserState(
                user_id=user_id,
                z_long=self.global_init_z.copy(),
                z_short=np.zeros(self.k, dtype=np.float32),
                reward_ma=0.0,
            )
            self._states[user_id] = state
        return self._states[user_id]

    def save_state(self, state: UserState) -> None:
        self._states[state.user_id] = state
    
    def persist(self):
        """Public method to force save to disk."""
        self._save()

