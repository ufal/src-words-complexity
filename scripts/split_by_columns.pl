#!/usr/bin/env perl
use strict;
use warnings;
use utf8;

binmode STDIN, ":utf8";

my @fhs = map {open my $fh, ">:utf8", $_; $fh} @ARGV;
while (my $line = <STDIN>) {
    chomp $line;
    my @cols = split /\t/, $line;
    if (scalar(@fhs) != scalar(@cols)) {
        printf STDERR "Number of output files on the command line must be the same as columns in the input file (%d<>%d).", scalar(@fhs), scalar(@cols);
        exit(1);
    }
    foreach my $i (0..$#fhs) {
        print {$fhs[$i]} $cols[$i]."\n"; 
    }
}
foreach my $fh (@fhs) {
    close $fh;
}
