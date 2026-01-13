#!/bin/sh

jobflag=$1

colldir=collect_${jobflag}
mkdir $colldir

for i in reaxff_${jobflag}_*/neff.bond \
	reaxff_${jobflag}_*/neff.work  \
	reaxff_${jobflag}_*/trajectory_sample*xyz \
	reaxff_${jobflag}_*/run.log \
	reaxff_${jobflag}_*/neff.log; do 
	echo $i
	f=`echo $i | sed -e "s/\//_/g"`;
	echo $f
	cp $i $colldir
done

zip -qr ${colldir}.zip $colldir

