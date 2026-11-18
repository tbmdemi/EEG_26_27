"""EEG preprocessing: raw preparation, behavioural scoring, filtering, ICA, epoching and QC."""
from .behaviour import score_trials, summarise
from .channels import bad_channel_scores
from .ica import ICAProgress
from .io import download_subject, ensure_utf8_tsv, fetch_metadata, has_recording, prepare_raw, remove_raw_recording
from .pipeline import PIPELINE_VERSION, preprocess, run, usable_subjects, write_log
from .qc import epoch_qc_plot, qc_plot
from .verify import save_epoch_qc, verify_epochs

__all__ = [
    'PIPELINE_VERSION', 'ICAProgress', 'bad_channel_scores', 'download_subject', 'ensure_utf8_tsv',
    'epoch_qc_plot', 'fetch_metadata', 'has_recording', 'prepare_raw', 'preprocess', 'qc_plot',
    'remove_raw_recording', 'run', 'save_epoch_qc', 'score_trials', 'summarise', 'usable_subjects',
    'verify_epochs', 'write_log',
]
