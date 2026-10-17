#!/bin/bash
# Tải ảnh chân dung cũ Việt Nam từ Flickr manhhai (KHÔNG CẦN API KEY).
# Dùng gallery-dl.
#
# Cài gallery-dl trước: pip install gallery-dl
#
# Sau khi tải xong, dùng filter_helper.py để lọc ảnh đạt chuẩn.

set -e

OUT_DIR="${1:-data/real_vn_old/raw}"
MAX_PHOTOS="${2:-500}"

mkdir -p "$OUT_DIR"

echo "Downloading photos from manhhai (Flickr)..."
echo "Output: $OUT_DIR"
echo "Max: $MAX_PHOTOS"

gallery-dl \
  --directory "$OUT_DIR" \
  --range "1-$MAX_PHOTOS" \
  --filename "manhhai_{id}.{extension}" \
  --sleep 1 \
  https://www.flickr.com/photos/13476480@N07/

echo "Done. Tổng số ảnh:"
find "$OUT_DIR" -type f -name "*.jpg" | wc -l
