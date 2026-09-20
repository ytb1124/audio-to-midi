"""HTTP Range로 원격 zip의 개별 멤버만 읽는 최소 구현.

E-GMD 전체 zip은 89.8GiB라 다 받는 데 몇 시간 걸린다. zip은 중앙 디렉터리가
파일 끝에 있고 멤버마다 오프셋이 기록돼 있으므로, Range 요청으로 필요한 wav
몇 개만 즉시 꺼낼 수 있다. 다운로드 완료를 기다리지 않고 검증하기 위한 도구.
"""
import io
import urllib.request
import zipfile


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url, block=1 << 20):
        self.url = url
        self.pos = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = off
        elif whence == io.SEEK_CUR:
            self.pos += off
        else:
            self.pos = self.size + off
        self.pos = max(0, min(self.pos, self.size))
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        end = min(self.pos + n, self.size) - 1
        if end < self.pos:
            return b""
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{end}"}
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except Exception:
                if attempt == 3:
                    raise
        self.pos += len(data)
        return data


def open_remote_zip(url):
    return zipfile.ZipFile(HttpRangeFile(url))
