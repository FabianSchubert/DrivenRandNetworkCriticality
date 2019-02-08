#! /bin/bash
bibfile="/home/fschubert/work/lit_base.bib"
newbibfile="driven_rand_net_notes.bib"
texfile="driven_rand_net_notes.tex"

for bibtag in `grep -o "\cite{[a-z_A-Z0-9,\ ]*}" $texfile | grep -o "[a-zA-Z]*_[0-9]*"`
do
	bibentry=`sed -n '/^@[a-z]*{'"$bibtag"'/,/^}/p' $bibfile`
	echo "$bibentry" >> $newbibfile
	echo "" >> $newbibfile
done
