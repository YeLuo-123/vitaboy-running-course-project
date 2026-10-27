"""Plot measured paired validation results, including every environment."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    run = args.run
    report = json.loads((run/'validation_report.json').read_text())
    seeds = json.loads((run/'plan.json').read_text())['final_new_seeds']
    def read(label):
        return json.loads((run/'results'/f'{label}.json').read_text())
    before = [read(f'baseline_final_transitions_{s}') for s in seeds]
    after = [read(f'candidate_final_transitions_{s}') for s in seeds]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, key, threshold, title in [
        (axes[0,0], 'stopped_speed_mps', .15, 'Settled planar speed / m/s'),
        (axes[0,1], 'turn_mean_bias_by_direction_rads', .08, 'Worst direction mean yaw bias / rad/s')]:
        def values(rows):
            v=[np.asarray(r[key]) for r in rows]
            return np.concatenate([np.abs(a).max(axis=1) if a.ndim==2 else a for a in v])
        b,a=values(before),values(after)
        upper=max(float(b.max()),float(a.max()),threshold)*1.12
        ax.scatter(b,a,s=21,alpha=.75,color='#244697')
        ax.plot([0,upper],[0,upper],color='#9aa5b4',lw=1)
        ax.axhline(threshold,color='#d36a18',ls='--',label='Pass threshold')
        ax.set(xlim=(0,upper),ylim=(0,upper),xlabel='Baseline',ylabel='Candidate',title=title)
        ax.grid(alpha=.18);ax.legend(fontsize=8)
    ax=axes[1,0]
    b=[r['completion']*100 for r in report['baseline_paired_terrain']]
    a=[r['completion']*100 for r in report['new_terrain']]
    x=np.arange(len(seeds));ax.bar(x-.18,b,.36,label='Baseline',color='#a4b2ce');ax.bar(x+.18,a,.36,label='Candidate',color='#244697')
    ax.axhline(95,color='#d36a18',ls='--');ax.set(xticks=x,xticklabels=[str(s) for s in seeds],ylim=(85,101),ylabel='Completion / %',title='Paired new terrain seeds: 128 trials each');ax.legend()
    ax=axes[1,1]
    checks=report['checks'];cpu=report['cpu_endurance']
    rows=[f"Validation checks passed: {sum(checks.values())} / {len(checks)}",
          f"Legacy terrain: {report['legacy_terrain_successes']} / 384 (baseline 374)",
          f"New terrain: {report['new_terrain_successes']} / 384",
          f"Paired baseline: {report['paired_baseline_terrain_successes']} / 384",
          f"CPU duration: {cpu['simulated_duration_s']:.1f} s",
          f"CPU whole-trial forward speed: {cpu['whole_trial_forward_speed_mps']:.3f} m/s",
          f"Result: {'PASS' if report['all_checks_passed'] else 'NOT PROMOTED'}"]
    ax.axis('off');ax.text(.01,.95,'\n\n'.join(rows),va='top',fontsize=11,color='#1c273a')
    fig.suptitle('Running stop/turn optimization: measured validation',fontsize=16)
    fig.savefig(run/'validation_comparison.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()
