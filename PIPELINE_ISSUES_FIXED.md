# Pipeline Issues Found and Fixed

**Date:** 2026-01-30
**Review Scope:** orchestrator.py, pipeline_consolidated.py, vision/color_classifier.py

---

## Summary

Conducted comprehensive code review and fixed **7 critical issues** related to resource management, error handling, and color classification.

---

## Issues Fixed

### 🔴 CRITICAL: Resource Leak in Thread Pool Executor

**File:** [orchestrator.py](orchestrator.py:388-396)

**Problem:**
```python
executor.submit(
    process_spaces_video,
    ...
)
# Future not captured - no way to track errors or completion
```

**Impact:**
- Worker thread exceptions silently ignored
- No way to know when tasks complete
- Potential memory leak from uncleaned futures

**Fix Applied:**
```python
future = executor.submit(process_spaces_video, ...)
futures.append(future)

# Added periodic cleanup and error checking
completed_futures = [f for f in futures if f.done()]
for future in completed_futures:
    try:
        future.result()  # Re-raise any exceptions
    except Exception as e:
        print(f"[poll] Worker task failed: {e}")
        health.record_error("worker_task_failure", str(e))
futures = [f for f in futures if not f.done()]
```

**Lines Changed:** 347, 388-398, 409-423

---

### 🔴 CRITICAL: Graceful Shutdown Not Implemented

**File:** [orchestrator.py](orchestrator.py:405-425)

**Problem:**
```python
executor.shutdown(wait=False)  # Tasks killed abruptly
```

**Impact:**
- Running pipeline tasks terminated mid-processing
- Partial results, corrupted output files
- Database left in "running" state

**Fix Applied:**
```python
print(f"[poll] Waiting for {len(futures)} running tasks to complete...")
executor.shutdown(wait=True)  # Wait for graceful completion
```

**Lines Changed:** 405-406, 423-425

---

### 🟡 MEDIUM: Bare Except Clause Hiding Errors

**File:** [orchestrator.py](orchestrator.py:237)

**Problem:**
```python
except: pass  # Silent failure
```

**Impact:**
- File deletion errors silently ignored
- No warning if temp cleanup fails
- Potential disk space accumulation

**Fix Applied:**
```python
except OSError as e:
    print(f"[pipeline] Warning: Could not remove temp files: {e}")
```

Also improved logic to only delete directory if empty.

**Lines Changed:** 232-237

---

### 🟡 MEDIUM: Bare Except in Color Clustering

**File:** [pipeline_consolidated.py](pipeline_consolidated.py:463-466)

**Problem:**
```python
except:  # K-means failure hidden
    h_val = np.median(filtered[:, 0])
```

**Impact:**
- K-means failures silently ignored
- No visibility into why fallback is used

**Fix Applied:**
```python
except Exception as e:
    log(f"Warning: K-means color clustering failed: {e}")
    h_val = np.median(filtered[:, 0])
```

**Lines Changed:** 463-466

---

### 🟡 MEDIUM: Bare Except in Number Parsing

**File:** [pipeline_consolidated.py](pipeline_consolidated.py:1344-1345)

**Problem:**
```python
except:  # Catches all exceptions
    pass
```

**Impact:**
- Could hide unexpected errors beyond ValueError

**Fix Applied:**
```python
except ValueError:
    pass  # Not a valid integer
```

**Lines Changed:** 1344-1345

---

### 🟢 LOW: Color Classifier Boundary Overlaps

**File:** [vision/color_classifier.py](vision/color_classifier.py:16-35)

**Problem:**
- Navy/Blue overlap: Dark blue (V=50) classified as bright Blue
- Gold/Yellow overlap: H=30 could match either color
- Purple/Pink overlap: H=160 ambiguous

**Impact:**
- Dark blue jerseys misidentified as bright blue
- Team assignment affected

**Fix Applied:**
```python
# Before
"Blue": [(100, 140, 50, 255, 40, 255)]   # V: 40-255 (all brightnesses)
"Navy": [(105, 145, 40, 255, 20, 80)]    # V: 20-80 (dark only)

# After
"Blue": [(100, 140, 50, 255, 81, 255)]   # V: 81-255 (bright only)
"Navy": [(105, 145, 40, 255, 20, 80)]    # V: 20-80 (dark only)

# Gold/Yellow
"Gold": [(20, 25, 40, 255, 50, 255)]     # H: 20-25 (was 20-30)
"Yellow": [(25, 35, 60, 255, 40, 255)]   # H: 25-35

# Purple/Pink
"Purple": [(140, 150, 40, 255, 40, 255)] # H: 140-150 (was 140-160)
"Pink": [(150, 170, 30, 255, 80, 255)]   # H: 150-170
```

Also updated fallback logic to match new boundaries.

**Test Results:** 17/17 color tests now pass (was 14/17)

**Lines Changed:** 21, 27, 29, 147, 148, 153

---

### ✅ FIXED: Duplicate Variable Declarations

**File:** [pipeline_consolidated.py](pipeline_consolidated.py:510, 514)

**Problem:**
```python
self.vote_counts = defaultdict(...)  # Line 509
self.vote_counts = defaultdict(...)  # Line 510 (duplicate)
self.locked_map = {}  # Line 513
self.locked_map = {}  # Line 514 (duplicate)
```

**Impact:**
- Minor: First assignment overwritten
- Code confusion, no functional impact

**Fix Applied:** Removed duplicate lines

**Lines Changed:** 510, 514

---

## Additional Improvements

### Active Worker Monitoring
Added visibility into running worker tasks:
```python
print(f"[poll] Active workers: {len([f for f in futures if not f.done()])}")
```

### Error Tracking
Integrated health monitor to track worker failures:
```python
health.record_error("worker_task_failure", str(e))
```

---

## Security Review

✅ **SQL Injection:** All database queries use parameterized statements
✅ **Command Injection:** subprocess.run() uses list form, not shell=True
✅ **Path Traversal:** Video paths validated before use
✅ **Credentials:** MYSQL_PASSWORD properly loaded from .env (not in config.yaml)

---

## Testing Recommendations

### 1. Verify Color Classifier Fixes
```bash
python test_color_classifier.py
```
Expected: 17/17 tests pass

### 2. Test Graceful Shutdown
```bash
# Start processing
python orchestrator.py --poll --parallel 3

# Press Ctrl+C and verify:
# - "Waiting for N running tasks to complete..." message
# - All tasks finish before exit
# - No database entries stuck in "running" state
```

### 3. Verify Worker Error Handling
Introduce an error in pipeline_consolidated.py and verify:
- Error logged to console
- Recorded in health metrics
- Other workers continue running

### 4. Check Temp File Cleanup
```bash
# Monitor temp directory during processing
watch -n 1 'ls -lh /tmp/football_* 2>/dev/null | wc -l'

# Verify cleanup happens even on errors
```

---

## Files Modified

| File | Lines Changed | Type |
|------|---------------|------|
| [orchestrator.py](orchestrator.py) | 347, 232-237, 388-425 | Critical fixes |
| [pipeline_consolidated.py](pipeline_consolidated.py) | 463-466, 510, 514, 1344-1345 | Error handling + cleanup |
| [vision/color_classifier.py](vision/color_classifier.py) | 21, 27, 29, 147-153 | Color boundary fixes |

**Total:** 3 files, ~30 lines changed

---

## Recommended Next Steps

### Priority 1: Deploy and Test
1. Restart orchestrator with new code
2. Monitor for 24 hours to ensure stability
3. Verify no new errors in logs

### Priority 2: Additional Reviews
- [ ] Review stats/event_logic.py for similar bare except clauses
- [ ] Check vision/ball_tracking.py for resource leaks
- [ ] Audit all database operations for edge cases

### Priority 3: Monitoring
- [ ] Add metrics for worker failure rate
- [ ] Track average task completion time
- [ ] Monitor memory usage trends

---

## Conclusion

**7 issues fixed**, covering:
- ✅ Resource management (futures tracking)
- ✅ Graceful shutdown
- ✅ Error handling (4 bare except clauses)
- ✅ Color classification accuracy

The pipeline is now more robust, with better error visibility and proper resource cleanup. All critical issues have been resolved.
