#!/usr/bin/env perl

use strict;
use warnings;
use utf8;

use List::MoreUtils qw/any/;

open my $in1, "<:utf8:gzip", $ARGV[0];
open my $in2, "<:utf8:gzip", $ARGV[1];
open my $in3, "<:utf8:gzip", $ARGV[2];

open my $out1, ">:utf8:gzip", $ARGV[3];
open my $out2, ">:utf8:gzip", $ARGV[4];
open my $out3, ">:utf8:gzip", $ARGV[5];

while (my $line1 = <$in1>) {
    my $line2 = <$in2>;
    my $line3 = <$in3>;
    next if (any {$_ =~ /^\s*$/} ($line1, $line2, $line3));
    print {$out1} $line1;
    print {$out2} $line2;
    print {$out3} $line3;
}
