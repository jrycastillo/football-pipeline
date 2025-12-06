import uuid
from collections import defaultdict

class GlobalRegistry:
    def __init__(self):
        # Maps (team, jersey_number) -> global_id
        self.jersey_registry = {}
        
        # Maps track_id -> global_id
        self.active_tracks = {}
        
        # Maps global_id -> { "team": ..., "number": ..., "track_ids": set() }
        self.global_identities = {}
        
        # Start Global IDs at a high number to avoid conflict with Track IDs
        self.next_global_id = 1000000

    def get_or_create_global_id(self, team, number, track_id):
        """
        Resolves the Global ID for a given track based on its team and jersey number.
        """
        if number == "Unknown":
            return self._register_unknown(track_id, team)

        # 1. Fuzzy Lookup: Team="Unknown" but Number is Known
        if team == "Unknown":
            # Check if this number exists in the registry for ANY team
            candidates = [k for k in self.jersey_registry.keys() if k[1] == number]
            if len(candidates) == 1:
                # Unambiguous match! Assume this is the player.
                # assumed_team = candidates[0][0]
                # print(f"[GlobalRegistry] Fuzzy Match: Track {track_id} (Unknown Team, #{number}) -> Assumed Team {assumed_team}")
                # We return the existing ID, but we don't necessarily update the registry key for (Unknown, 10)
                # unless we want to cache it. Let's just return the ID.
                return self.jersey_registry[candidates[0]]
            
            # If ambiguous or new, fall through to create/lookup (Unknown, 10)

        key = (team, number)
        
        if key in self.jersey_registry:
            # Re-Identification: Existing player
            global_id = self.jersey_registry[key]
            self.active_tracks[track_id] = global_id
            self.global_identities[global_id]["track_ids"].add(track_id)
            return global_id
        else:
            # 2. Merge Logic: Check if (Unknown, Number) exists
            # If we are registering (Red, 10), but (Unknown, 10) exists, we should claim it.
            unknown_key = ("Unknown", number)
            if unknown_key in self.jersey_registry:
                # Merge!
                global_id = self.jersey_registry[unknown_key]
                print(f"[GlobalRegistry] Merging Identity: Found (Unknown, #{number}) -> Updating to ({team}, #{number}) (Global ID {global_id})")
                
                # Link new key
                self.jersey_registry[key] = global_id
                
                # Update metadata
                self.global_identities[global_id]["team"] = team
                self.global_identities[global_id]["track_ids"].add(track_id)
                self.active_tracks[track_id] = global_id
                
                # Optional: Remove unknown_key? No, keep it as an alias.
                return global_id

            # New Player
            global_id = self._create_new_identity(team, number)
            self.jersey_registry[key] = global_id
            self.active_tracks[track_id] = global_id
            self.global_identities[global_id]["track_ids"].add(track_id)
            print(f"[GlobalRegistry] New Identity: Track {track_id} assigned to {team} #{number} (Global ID {global_id})")
            return global_id

    def _create_new_identity(self, team, number):
        global_id = self.next_global_id
        self.next_global_id += 1
        
        self.global_identities[global_id] = {
            "team": team,
            "number": number,
            "track_ids": set()
        }
        return global_id

    def _register_unknown(self, track_id, team):
        # For unknown players, we treat each track as a unique identity 
        # (unless we have advanced tracking logic, which we don't yet).
        # We check if we already assigned a GID to this track (unlikely in batch, but good practice).
        if track_id in self.active_tracks:
            return self.active_tracks[track_id]
            
        # Create a new "Unknown" identity
        global_id = self.next_global_id
        self.next_global_id += 1
        
        self.active_tracks[track_id] = global_id
        self.global_identities[global_id] = {
            "team": team,
            "number": "Unknown",
            "track_ids": {track_id}
        }
        return global_id
