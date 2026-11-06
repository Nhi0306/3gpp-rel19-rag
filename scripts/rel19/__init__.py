"""3GPP Rel-19 pipeline package.

Mỗi giai đoạn nằm trong một module riêng:

    rel19/paths.py     - đường dẫn, hằng số, URL
    rel19/common.py    - HTTP, JSON, log, parse tên file
    rel19/manifest.py  - crawl status report + FTP listings
    rel19/download.py  - tải zip
    rel19/extract.py   - giải nén docx
    rel19/parse.py     - docx -> sections/blocks JSON
    rel19/chunk.py     - chunk theo heading rồi theo token
    rel19/embed.py     - embed + upsert Qdrant
    rel19/update.py    - kiểm tra spec mới và cập nhật tăng dần
    rel19/stats.py     - thống kê tiến độ
"""
