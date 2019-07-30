#!/usr/bin/env perl

use strict;
use warnings;
use utf8;

my %addresses;

open my $bl_f, "<", $ARGV[0];
while (my $line = <$bl_f>) {
    chomp $line;
    my ($path, $fileno, $lineno) = ($line =~ /^(.*)\/(\d+)-(\d+)$/);
    $addresses{$path}{$fileno}{$lineno} = 1;
    # print STDERR "$path $fileno $lineno\n";
}
close $bl_f;

while (my $line = <STDIN>) {
    chomp $line;
    my ($path, $fileno) = ($line =~ /^(.*)\/[^\/]*\.(\d+)\.\w+$/);
    # print STDERR "$path $fileno\n";
    my %lines_to_delete = ();
    if ($addresses{$path}{$fileno}) {
        %lines_to_delete = %{$addresses{$path}{$fileno}};
    }
    my $lineno = 1;
    open my $f, "<", $line;
    while (<$f>) {
        print $_ if (!$lines_to_delete{$lineno});
        $lineno++;
    }
}
