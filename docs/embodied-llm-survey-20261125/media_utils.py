"""Small ISO BMFF reader: inspect the video track, never an audio track."""
import struct


def mp4_metadata(path):
    data = path.read_bytes()

    def boxes(start, end):
        while start + 8 <= end:
            size, kind = struct.unpack_from('>I4s', data, start)
            header = 8
            if size == 1:
                size = struct.unpack_from('>Q', data, start + 8)[0]
                header = 16
            if size == 0:
                size = end - start
            if size < header or start + size > end:
                raise ValueError('Invalid MP4 box')
            yield kind, start + header, start + size
            start += size

    def descendants(start, end):
        for kind, a, b in boxes(start, end):
            yield kind, a, b
            if kind in (b'mdia', b'minf', b'stbl'):
                yield from descendants(a, b)

    tracks = []
    for kind, a, b in boxes(0, len(data)):
        if kind != b'moov':
            continue
        for kind, a, b in boxes(a, b):
            if kind != b'trak':
                continue
            track = {}
            for kind, a, b in descendants(a, b):
                if kind == b'hdlr':
                    track['handler'] = data[a+8:a+12].decode('ascii')
                elif kind == b'mdhd':
                    offset = a + (20 if data[a] == 1 else 12)
                    scale = struct.unpack_from('>I', data, offset)[0]
                    ticks = struct.unpack_from('>Q' if data[a] == 1 else '>I', data, offset+4)[0]
                    track.update(duration_s=ticks/scale, timescale=scale)
                elif kind == b'stsz':
                    track['sample_count'] = struct.unpack_from('>I', data, a+8)[0]
                elif kind == b'stts':
                    count = struct.unpack_from('>I', data, a+4)[0]
                    track['sample_timing'] = [list(struct.unpack_from('>II', data, a+8+i*8)) for i in range(count)]
                elif kind == b'tkhd':
                    track['width'], track['height'] = [n/65536 for n in struct.unpack_from('>II', data, b-8)]
                elif kind == b'stsd':
                    track['codec'] = data[a+12:a+16].decode('ascii')
            tracks.append(track)
    video = [t for t in tracks if t.get('handler') == 'vide']
    assert len(video) == 1, tracks
    result = video[0]
    result['avc1_present'] = result.get('codec') == 'avc1'
    timing = result['sample_timing']
    assert sum(n for n, _ in timing) == result['sample_count']
    result['constant_frame_rate'] = len({delta for _, delta in timing}) == 1
    result['fps'] = result['timescale']/timing[0][1] if result['constant_frame_rate'] else None
    return result
