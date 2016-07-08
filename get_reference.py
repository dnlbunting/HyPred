#!/usr/bin/env python
import pandas as pd
import re

import Bio.SeqIO

markers_1 = pd.read_csv("markers_1.txt")
loci = [re.match("(\S+)_(\S+)", x).groups() for x in markers_1['markers']]
loci = [("PST130_"+x[0], x[1]) for x in loci]

f = open("loci.txt", 'w')
for x in loci_2:
    f.write(x[0]+"\t"+x[1]+"\n")
f.close()


ref_file = Bio.SeqIO.parse("PST130_contigs.fasta", 'fasta')
ref = [x for x in ref_file]
reference = {x.name:x for x in ref}

input = open("loci.txt", 'r')
output = open("reference.txt", 'w')

output.write("markers\tref\n")
for line in input:
    line = line.rstrip()
    contig, pos1 = line.split("\t")
    output.write(contig[7:]+'_'+pos1 + "\t" + reference[contig][int(pos1)-1] + "\n")
    
input.close()
output.close()