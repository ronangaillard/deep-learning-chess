from bitarray import bitarray


NEXT_POS = bitarray(0)

with open("data.bin") as binfile:
    NEXT_POS.fromfile(binfile, 768/8)
    NEXT_POS.fromfile(binfile, 768/8)

print NEXT_POS
print "len", NEXT_POS.length()
