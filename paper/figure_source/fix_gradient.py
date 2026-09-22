#!/usr/bin/env python3
"""Replace only the MoE banner's shading-pattern fill by direct vector shading.

The source PDF is unchanged except PDF serialization and this equivalent fill.
This avoids the Quartz nested-Form shading-pattern coordinate incompatibility.
"""
import sys
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, ArrayObject, FloatObject, DictionaryObject, DecodedStreamObject
src,dst=map(Path,sys.argv[1:3])
r=PdfReader(src)
w=PdfWriter(); w.clone_document_from_reader(r)
p=w.pages[0]
s=p.get_contents().get_data()
start=s.index(b'q /Pattern cs /p16 scn /a2 gs\n')
end=s.index(b'\nQ\n',start)+3
old=s[start:end]
assert b'102 2.5 m B' in old
assert old.count(b' B') == 1
# Retain the exact original rounded-rectangle path and white stroke settings.
path_start=old.index(b'4 M ')+4
path_end=old.index(b' B',path_start)
path=old[path_start:path_end]
style=old[old.index(b'1 1 1 RG'):path_start]
replacement=(b'q /a2 gs\n'+style+b'q\n'+path+b' W n\n/rvMoEBanner sh\nQ\n'+path+b' S\nQ\n')
s=s[:start]+replacement+s[end:]
shading=DictionaryObject(dict(p['/Resources']['/Pattern']['/p16']['/Shading']))
# Native content coordinates; the axial gradient is horizontal.
shading[NameObject('/Coords')]=ArrayObject([FloatObject(v) for v in [99,2.5,228,2.5]])
p['/Resources']['/Shading'][NameObject('/rvMoEBanner')]=w._add_object(shading)
stream=DecodedStreamObject();stream.set_data(s);p[NameObject('/Contents')]=w._add_object(stream)
w.write(dst)
print(dst)
