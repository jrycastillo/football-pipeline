# Ground-Truth Events Format Specification

This document details the JSON schema for human-annotated event-level ground-truth data. This data is used to evaluate the AI pipeline's event-level precision, recall, and F1 score.

## JSON Schema

The ground-truth input must be a JSON array of objects. Each object represents a single human-annotated match event and contains the following fields:

| Field | Type | Required | Description / Constraints |
|---|---|---|---|
| `type` | string | **Yes** | The type of the event. Must be exactly one of: `"pass"`, `"shot"`, `"goal"`, `"tackle"`, `"foul"`, `"interception"`, `"touch"`. |
| `time_s` | float | **Yes** | The event timestamp in seconds from the start of the video (e.g. `124.5`). |
| `team` | string | No | The team associated with the primary actor (e.g. `"Red"`, `"White"`). |
| `player` | integer | No | The jersey number of the player performing or receiving the action. |

---

## Worked Example

Below is a complete, valid example of an event ground-truth file (`ground_truth_events.json`):

```json
[
  {
    "type": "touch",
    "time_s": 12.4,
    "team": "Red",
    "player": 10
  },
  {
    "type": "pass",
    "time_s": 14.2,
    "team": "Red",
    "player": 10
  },
  {
    "type": "interception",
    "time_s": 15.8,
    "team": "White",
    "player": 4
  },
  {
    "type": "tackle",
    "time_s": 22.1,
    "team": "Red",
    "player": 7
  },
  {
    "type": "foul",
    "time_s": 23.5,
    "team": "White",
    "player": 8
  },
  {
    "type": "shot",
    "time_s": 45.0,
    "team": "Red",
    "player": 9
  },
  {
    "type": "goal",
    "time_s": 45.8,
    "team": "Red",
    "player": 9
  }
]
```
