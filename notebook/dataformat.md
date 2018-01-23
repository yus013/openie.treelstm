# Data Format

## Summary
* data/[train, dev, test]/[sent, arb]


## arb.txt
* sentence index start from 1
* sentence appears in ascending order
* arb index start from 0
* arb is extracted from tokenized sentence
* differen parts are divided by space
* words are divided by comma

sent_id a0,a1,...an r0,r1,...rn b0,b1,...,bn

## label.txt
* sentence index start from 1
* sentence appears in ascending order
* label appears in the order with respect to arb
* 0 stands for false while 1 stands for true

sent_id 1/0

## sent.txt
* raw sentences

