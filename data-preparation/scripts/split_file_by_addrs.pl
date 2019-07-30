#!/usr/bin/env perl

use strict;
use warnings;
use utf8;

use File::Basename;
use File::Spec::Functions;
use File::Path qw/make_path/;

open my $addr_fh, "<:utf8:gzip", $ARGV[0];
my $patt = $ARGV[1];
my $out_dir = $ARGV[2];

my $last_file;
my $last_fh;
while (my $line = <STDIN>) {
    my $addr = <$addr_fh>;
    chomp $addr;
    $addr =~ s/-s\d+$//;
    $addr =~ s/\{[^}]*\}/$patt/;
    if (!defined $last_file || $last_file ne $addr) {
        if ($last_fh) {
            close $last_fh;
        }
        $last_file = $addr;
        my ($name, $path) = fileparse($addr);
        make_path catfile($out_dir, $path);
        my $full_path = catfile($out_dir, $path, $name);
        print STDERR $full_path."\n";
        open $last_fh, ">:utf8", $full_path;
    }
    print {$last_fh} $line;
}

close $last_fh;
close $addr_fh;
