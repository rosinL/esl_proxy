#!/usr/bin/env python3
import sys, re

def parse_dag(filename):
    tasks = {}
    preds = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith('TASK '):
                tid = int(line.split()[1].rstrip(':'))
                m = re.search(r'type=(\d+)\s+mode=(\d+)\s+count=(\d+)\s+dur=(\d+)\s+tc=(\d+)\s+sc=(\d+)', line)
                sm = re.search(r'scalar=\[([^\]]*)\]', line)
                tasks[tid] = {
                    'type': m.group(1), 'mode': m.group(2), 'count': m.group(3),
                    'dur': m.group(4), 'tc': m.group(5), 'sc': m.group(6),
                    'scalar': sm.group(1) if sm else ''
                }
            elif line.startswith('PRED '):
                tid = int(line.split()[1].rstrip(':'))
                m = re.search(r'cnt=(\d+)\s*\[([^\]]*)\]', line)
                cnt = int(m.group(1)) if m else 0
                preds_str = m.group(2).strip() if m else ''
                preds_list = sorted([int(x) for x in preds_str.split(',') if x.strip()]) if preds_str else []
                preds[tid] = {'cnt': cnt, 'preds': preds_list}
    return tasks, preds

def compare(orig_file, pipe_file):
    ot, op = parse_dag(orig_file)
    pt, pp = parse_dag(pipe_file)

    errors = []

    if set(ot.keys()) != set(pt.keys()):
        errors.append(f"Task count mismatch: orig={len(ot)} pipe={len(pt)}")
        errors.append(f"  Only in orig: {set(ot.keys()) - set(pt.keys())}")
        errors.append(f"  Only in pipe: {set(pt.keys()) - set(ot.keys())}")

    task_mismatches = 0
    for tid in sorted(ot.keys()):
        if tid not in pt:
            errors.append(f"TASK {tid} missing in pipe")
            continue
        o = ot[tid]
        p = pt[tid]
        for key in ['type', 'mode', 'count', 'dur', 'tc', 'sc', 'scalar']:
            if o[key] != p[key]:
                errors.append(f"TASK {tid} field '{key}': orig={o[key]} pipe={p[key]}")
                task_mismatches += 1

    pred_mismatches = 0
    for tid in sorted(op.keys()):
        if tid not in pp:
            errors.append(f"PRED {tid} missing in pipe")
            pred_mismatches += 1
            continue
        if op[tid]['cnt'] != pp[tid]['cnt']:
            errors.append(f"PRED {tid} cnt: orig={op[tid]['cnt']} pipe={pp[tid]['cnt']}")
            pred_mismatches += 1
            continue
        if op[tid]['preds'] != pp[tid]['preds']:
            errors.append(f"PRED {tid} set: orig={op[tid]['preds']} pipe={pp[tid]['preds']}")
            pred_mismatches += 1

    if errors:
        print(f"FAILED: {task_mismatches} task mismatches, {pred_mismatches} pred mismatches")
        for e in errors[:30]:
            print(f"  {e}")
        if len(errors) > 30:
            print(f"  ... and {len(errors) - 30} more")
        return 1
    else:
        print(f"PASSED: {len(ot)} tasks, {len(op)} preds, all match")
        return 0

if __name__ == '__main__':
    sys.exit(compare(sys.argv[1], sys.argv[2]))
