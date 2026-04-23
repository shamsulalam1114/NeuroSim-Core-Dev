
import warnings
from pathlib import Path

try:
    from bids import BIDSLayout
    _PYBIDS_OK = True
except ImportError:
    BIDSLayout = None  # keep name in module namespace for mock patching
    _PYBIDS_OK = False


def load_bids_layout(bids_dir, derivatives=True):
    p = Path(bids_dir)
    if not p.exists():
        raise FileNotFoundError(f"BIDS directory not found: {bids_dir}")
    if not _PYBIDS_OK:
        raise ImportError("pybids is required. Install with: pip install pybids")
    return BIDSLayout(str(p), derivatives=derivatives)


def get_bold_files(layout, subject, task=None, space="MNI152NLin2009cAsym", res="2", desc="preproc"):
    # standard fMRIPrep output space and resolution
    filters = {"subject": subject, "suffix": "bold",
                "extension": [".nii", ".nii.gz"], "space": space, "desc": desc}
    if task is not None:
        filters["task"] = task
    if res is not None:
        filters["res"] = res
    return layout.get(**filters, return_type="filename")


def get_confound_files(layout, subject, task=None):
    # fMRIPrep confounds TSV — 24-parameter HMP + derivatives + WM/CSF
    filters = {"subject": subject, "suffix": "timeseries",
                "extension": ".tsv", "desc": "confounds"}
    if task is not None:
        filters["task"] = task
    return layout.get(**filters, return_type="filename")


def collect_subject_file_index(layout, task=None, space="MNI152NLin2009cAsym", res="2"):
    # build {sub_id: {"bold": [...], "confounds": [...], "n_runs": int}}
    index = {}
    for sub in sorted(layout.get_subjects()):
        bold = get_bold_files(layout, sub, task=task, space=space, res=res)
        confounds = get_confound_files(layout, sub, task=task)
        if not bold:
            warnings.warn(f"No BOLD files for subject '{sub}'.", UserWarning, stacklevel=2)
            continue
        index[sub] = {"bold": bold, "confounds": confounds, "n_runs": len(bold)}
    return index
