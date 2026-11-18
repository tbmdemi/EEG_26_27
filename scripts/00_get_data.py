"""Tải dữ liệu oddball đã tiền xử lý (data v2) vào đúng chỗ trong repo — chạy một lần là xong.

    python scripts/00_get_data.py                 # tải từ Google Drive, kiểm tra, giải nén
    python scripts/00_get_data.py --zip FILE.zip  # đã tải zip bằng trình duyệt -> chỉ kiểm tra + giải nén
    python scripts/00_get_data.py --check         # chỉ kiểm tra dữ liệu đang có

Chạy được cả trên Colab:  !python scripts/00_get_data.py

Kết quả: data/derivatives/preproc/ chứa sub-XX_task-oddball_epo.fif (+ ICA, summary, preproc_log.csv, config.json).
Chỉ dùng subject có excluded_subject = False trong preproc_log.csv (hoặc eegproj.preprocessing.usable_subjects).
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# ---- Bản phát hành hiện tại (PM cập nhật khi có bản mới) ----------------------------------------
RELEASE = {
    'name': 'preproc_v2_20261118.zip',
    'gdrive_file_id': '1hXdIGzBk8p1QziEy3DARKdihhgpi95S9',
    'folder_url': 'https://drive.google.com/drive/folders/1sqrYetZH3xr_eFbH2Cce8V_iwulI44KR',
    'sha256': 'c53f040786b6c23a9b5d135a31f0bef966088018434409a1ca44a2e944c7a6ef',
    'n_usable': 35,
    'n_bytes': 2_636_326_964,
}
# ---------------------------------------------------------------------------------------------------


def find_root():
    here = Path(__file__).resolve()
    for p in (here.parent, *here.parents, Path.cwd(), *Path.cwd().parents):
        if (p / 'configs' / 'config.yaml').exists():
            return p
    sys.exit('Không tìm thấy configs/config.yaml — hãy chạy script trong repo EEG_26_27.')


def derivatives_dir(root):
    try:
        import yaml
        cfg = yaml.safe_load((root / 'configs' / 'config.yaml').read_text(encoding='utf-8'))
        return root / cfg['paths']['derivatives']
    except Exception:
        return root / 'data' / 'derivatives' / 'preproc'


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def download(dest_zip):
    try:
        import gdown  # noqa: F401
    except ImportError:
        print('Cài gdown ...')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'], check=True)
    import gdown
    print(f"Tải {RELEASE['name']} (~{RELEASE['n_bytes'] / 1e9:.1f} GB) từ Google Drive ...")
    out = gdown.download(id=RELEASE['gdrive_file_id'], output=str(dest_zip), quiet=False, resume=True)
    if not out or not Path(dest_zip).exists():
        sys.exit('Tải thất bại (Google Drive có thể đang giới hạn lượt tải).\n'
                 f"Hãy tải {RELEASE['name']} bằng trình duyệt tại {RELEASE['folder_url']}\n"
                 f'rồi chạy: python scripts/00_get_data.py --zip <đường dẫn file zip>')


def usable_from_log(out):
    import pandas as pd
    log = pd.read_csv(out / 'preproc_log.csv')
    ok = log[~log.excluded_subject.fillna(True).astype(bool)]
    return list(ok.subject)


def check(out, deep=False):
    """Kiểm tra: đủ file epo cho mọi subject dùng được, metadata đúng phiên bản v2."""
    if not (out / 'preproc_log.csv').exists():
        return False, 'chưa có dữ liệu'
    subjects = usable_from_log(out)
    missing = [s for s in subjects if not (out / f'{s}_task-oddball_epo.fif').exists()]
    if missing:
        return False, f'thiếu file epo: {missing}'
    if len(subjects) != RELEASE['n_usable']:
        return False, f"có {len(subjects)} subject dùng được, mong đợi {RELEASE['n_usable']}"
    if deep:
        import mne
        ep = mne.read_epochs(out / f'{subjects[0]}_task-oddball_epo.fif', preload=False, verbose='error')
        if list(ep.metadata.columns) != ['subject', 'onset_s', 'block', 'rt'] or ep.info['sfreq'] != 250:
            return False, 'file epo không phải bản v2'
    return True, f'{len(subjects)} subject dùng được'


def set_aside_old(out, names_in_zip):
    """Chuyển file dữ liệu cũ (không thuộc bản phát hành này) sang _old/ — không xoá."""
    moved = []
    for f in out.iterdir():
        if f.is_file() and f.suffix in ('.fif', '.json', '.csv') and f.name not in names_in_zip:
            (out / '_old').mkdir(exist_ok=True)
            shutil.move(str(f), str(out / '_old' / f.name))
            moved.append(f.name)
    return moved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zip', type=Path, help='file zip đã tải sẵn (bỏ qua bước tải)')
    ap.add_argument('--check', action='store_true', help='chỉ kiểm tra dữ liệu hiện có')
    ap.add_argument('--keep-zip', action='store_true', help='giữ lại file zip sau khi giải nén')
    ap.add_argument('--force', action='store_true', help='tải và giải nén lại dù dữ liệu đã đủ')
    args = ap.parse_args()

    root = find_root()
    out = derivatives_dir(root)
    out.mkdir(parents=True, exist_ok=True)

    ok, msg = check(out, deep=True)
    if args.check:
        print(('OK: ' if ok else 'CHƯA ĐỦ: ') + msg + f'  ({out})')
        sys.exit(0 if ok else 1)
    if ok and not args.force:
        print(f'Dữ liệu đã đủ ({msg}) tại {out}. Không cần tải lại (dùng --force nếu muốn).')
        return

    zip_path = args.zip or (root / 'data' / RELEASE['name'])
    if not args.zip and not (zip_path.exists() and zip_path.stat().st_size == RELEASE['n_bytes']):
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        download(zip_path)

    print('Kiểm tra sha256 ...')
    digest = sha256sum(zip_path)
    if digest != RELEASE['sha256']:
        sys.exit(f'Sai sha256 ({digest[:12]}… ≠ {RELEASE["sha256"][:12]}…): file tải về bị lỗi hoặc không đúng bản. '
                 'Xoá file zip và chạy lại.')

    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        moved = set_aside_old(out, names)
        if moved:
            print(f'Đã chuyển {len(moved)} file cũ sang {out / "_old"}')
        print(f'Giải nén {len(names)} file vào {out} ...')
        z.extractall(out)

    ok, msg = check(out, deep=True)
    if not ok:
        sys.exit(f'Giải nén xong nhưng kiểm tra thất bại: {msg}')
    if not args.zip and not args.keep_zip:
        zip_path.unlink()
    print(f'XONG: {msg}. Dữ liệu tại {out}')
    print('Đọc thử:  mne.read_epochs("data/derivatives/preproc/sub-01_task-oddball_epo.fif")')


if __name__ == '__main__':
    main()
