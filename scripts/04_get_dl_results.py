"""R5: Tải outputs deep learning đã chạy sẵn vào results/, không cần train lại (thay cho scripts/04_train_dl.py).

    python scripts/04_get_dl_results.py                 # tải từ Google Drive, kiểm tra sha256, giải nén
    python scripts/04_get_dl_results.py --zip FILE.zip  # đã tải zip bằng trình duyệt: chỉ kiểm tra + giải nén
    python scripts/04_get_dl_results.py --force         # ghi đè cả file trong results/ đang khác bản phát hành

Nội dung: eegnet/shallow x within/cross x full/early, task tgt_std, 35 subject, data v2 (code commit fc0cfaf).
Kết quả: results/checkpoints/, results/predictions/, results/metrics/, results/dl_runs/, results/cv_cross_subject_folds.csv.
"""
import argparse
import hashlib
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

from eegproj.config import find_root

# Bản phát hành hiện tại (R5 cập nhật khi có bản mới)
RELEASE = {
    'name': 'r5_dl_results_20261120_fc0cfaf.zip',
    'gdrive_file_id': '1YTt4LVyr9dpk5J7_TzHkUlY3ab7xq_lI',
    'sha256': 'a02226ee2308c0e6ef316c0be700d746927295607cddd4779391f18773f86e9a',
    'n_bytes': 285_260_487,
}


def download(dest_zip):
    try:
        import gdown
    except ImportError:
        print('Cài gdown ...')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'], check=True)
        import gdown
    print(f"Tải {RELEASE['name']} (~{RELEASE['n_bytes'] / 1e6:.0f} MB) từ Google Drive ...")
    if not gdown.download(id=RELEASE['gdrive_file_id'], output=str(dest_zip), quiet=False, resume=True):
        sys.exit('Tải thất bại (Google Drive có thể đang giới hạn lượt tải).\n'
                 f"Tải bằng trình duyệt: https://drive.google.com/file/d/{RELEASE['gdrive_file_id']}/view\n"
                 'rồi chạy: python scripts/04_get_dl_results.py --zip <đường dẫn file zip>')


def compare(z, root):
    """So từng file trong zip với file trên máy (theo CRC): trả về (còn thiếu, đang khác nội dung)."""
    missing, differ = [], []
    for info in z.infolist():
        path = root / info.filename
        if info.is_dir():
            continue
        if not path.exists():
            missing.append(info.filename)
        elif zlib.crc32(path.read_bytes()) != info.CRC:
            differ.append(info.filename)
    return missing, differ


def install(zip_path, root, force):
    print('Kiểm tra sha256 ...')
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    if digest != RELEASE['sha256']:
        sys.exit(f"Sai sha256 ({digest[:12]} khác {RELEASE['sha256'][:12]}): file lỗi hoặc không đúng bản phát hành.")

    with zipfile.ZipFile(zip_path) as z:
        if any(not name.startswith('results/') for name in z.namelist()):
            sys.exit('Zip có file nằm ngoài results/, dừng lại.')
        missing, differ = compare(z, root)
        if differ and not force:
            # không ghi đè kết quả người dùng tự chạy khi chưa được phép
            sys.exit(f'{len(differ)} file trong results/ đang khác bản phát hành (có thể do bạn tự train), '
                     f'ví dụ: {differ[:3]}.\nSao lưu chúng rồi chạy lại với --force để ghi đè.')
        if not missing and not differ:
            print(f"results/ đã có đủ {RELEASE['name']}, không cần giải nén.")
            return
        print(f'Giải nén {len(missing)} file mới, ghi đè {len(differ)} file vào {root / "results"} ...')
        z.extractall(root, members=missing + differ)
        if any(compare(z, root)):
            sys.exit('Giải nén xong nhưng vẫn có file thiếu hoặc sai nội dung.')
    print('XONG. Đọc checkpoint: eegproj.models.dl.load_checkpoint; xem bảng: python scripts/04_train_dl.py --summary')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zip', type=Path, help='file zip đã tải sẵn (bỏ qua bước tải)')
    ap.add_argument('--force', action='store_true', help='ghi đè file trong results/ đang khác bản phát hành')
    args = ap.parse_args()

    root = find_root(Path(__file__).resolve().parent)
    zip_path = args.zip or root / 'data' / RELEASE['name']
    if not args.zip:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        download(zip_path)
    try:
        install(zip_path, root, args.force)
    finally:
        if not args.zip:  # chỉ xoá zip do script tự tải
            zip_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
