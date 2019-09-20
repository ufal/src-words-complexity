#!/usr/bin/env perl
use strict;
use warnings;
use utf8;

use Graph::Undirected;
use Data::Printer;
use List::Util qw/any all/;
use Getopt::Long;
use IO::Handle;

binmode STDOUT, ":utf8";
STDOUT->autoflush(1);


my @SIDES_ORDER = qw/src mt pe/;
my @ALI_ORDER = qw/src-mt src-pe pe-mt/;

sub load_stopwords {
    my ($path) = @_;
    return undef if (!$path);
    my $stopwords = {};
    open my $sw_f, "<:utf8", $path;
    while (<$sw_f>) {
        chomp $_;
        $stopwords->{$_} = 1;
    }
    close $sw_f;
    return $stopwords;
}

my $stopword_path;
my $out_components_path;
GetOptions(
    "stopwords=s" => \$stopword_path,
    "out-components=s" => \$out_components_path,
);
my $STOPWORDS = load_stopwords($stopword_path);

sub extract_from_sent_triple {
    my $token_index = tokenize_parts(@_[0..1]);
    my $ali_comps = find_aligned_components(@_[2..4]);
    return ($ali_comps, $token_index);
}

# INPUT: 3 token alignment strings representing 3 alignments between tokens (as indices) in corresponding {src, mt, pe} sentence
# OUTPUT: a list of aligned components represented as a hash indexed by {src, mt, pe} and with a list of token indices as values
# Take alignment strings and find connected components in them. Each component is returned as three lists of token indices, corresponding to {src, mt, pe}.
sub find_aligned_components {
    my @sent_aligns = @_;
    chomp $_ foreach (@sent_aligns);
    my %sent_map = map {$ALI_ORDER[$_] => [split / /, $sent_aligns[$_]]} 0..$#sent_aligns;
    my $g = Graph::Undirected->new();
    foreach my $align_name (keys %sent_map) {
        my ($an1, $an2) = split /-/, $align_name;
        foreach my $align_pair (@{$sent_map{$align_name}}) {
            my ($a1, $a2) = split /-/, $align_pair;
            $g->add_edge($an1.":".$a1, $an2.":".$a2)
        }
    }
    my @comps = sort {(join ",", $a) cmp (join ",", $b)} $g->connected_components;
    my @comps_split_to_sides = map {
        my %parts_hash = ();
        foreach my $token_label (@$_) {
            my ($part_name, $token_idx) = split /:/, $token_label;
            my $part_comp = $parts_hash{$part_name};
            if (!defined $part_comp) {
                $part_comp = [ $token_idx ];
                $parts_hash{$part_name} = $part_comp;
            }
            else {
                push @$part_comp, $token_idx;
            }
        }
        foreach my $part_name (keys %parts_hash) {
            $parts_hash{$part_name} = [ sort {$a <=> $b} @{$parts_hash{$part_name}} ];
        }
        \%parts_hash
    } @comps;
    return \@comps_split_to_sides;
}

# INPUT: 2 bitext sentences (src-mt, src-pe) as a list of strings
# OUTPUT: a hash indexed by {src, mt, pe} with lists of tokens as values
# Split the bitexts (at tabs) and tokenize each side of the sentence triple (at spaces).
sub tokenize_parts {
    my @bitext_lines = @_;
    @bitext_lines = map {chomp $_; split /\t/, $_} @bitext_lines;
    my @sides_sent = @bitext_lines[(1,2,5)];
    my %index = ();
    foreach my $side_idx (0..$#sides_sent) {
        my @sent_tokens = split / /, $sides_sent[$side_idx];
        $index{$SIDES_ORDER[$side_idx]} = \@sent_tokens;
    }
    return \%index;
}

sub is_token_content {
    my ($token) = @_;
    return 0 if ($token eq "|");
    my @token_parts = split /\|/, $token;
    return not any {  
        $_ =~ /^[.,:\|?_+-=()"'“„”&%><–—\]\[\/\\]+$/ ||
        $STOPWORDS->{$_}
    } @token_parts;
}

sub tokens_equal {
    my ($tokens1, $tokens2) = @_;
    return 0 if (scalar @$tokens1 != scalar @$tokens2);
    return all {
        my $i = $_;
        my @token1_parts = split /\|/, $tokens1->[$i];
        my @token2_parts = split /\|/, $tokens2->[$i];
        # if intersection of tokens' parts is not empty
        my %all_tokens_h;
        $all_tokens_h{$_}++ foreach (@token1_parts, @token2_parts);
        any {$all_tokens_h{$_} > 1} keys %all_tokens_h
    } (0..$#$tokens1);
}

sub filter_ali_comps {
    my ($ali_comps, $token_index) = @_;
    
    # constraints on source language phrases
    # filter out all components longer than one src word
    my @new_ali_comps = grep {
        my $comp = $_;
        ( defined $comp->{src} && scalar @{$comp->{src}} == 1)
    } @$ali_comps;
    #print STDERR "ALI_COMPS1: ".scalar(@new_ali_comps)."\n";
    
    # constraints on target language phrases - each side individually
    # filter out all components that contain only stopwords or punctuation on the target side (mt or pe)
    @new_ali_comps = grep {
        my $comp = $_;
        ( all {
            my $side_name = $_;
            my @tokens = indexes_to_tokens($comp->{$side_name}, $token_index->{$side_name});
            (@tokens && all {is_token_content($_)} @tokens) ? 1 : 0;
        } qw/mt pe/ )
    } @new_ali_comps;
    #print STDERR "ALI_COMPS2: ".scalar(@new_ali_comps)."\n";
    
    # filter out phrases that are the same in both target sides (excluding aux tokens, e.g. stopwords, punctuation)
    @new_ali_comps = grep {
        my $comp = $_;
        my @mt_tokens = indexes_to_tokens($comp->{mt}, $token_index->{mt});
        my @pe_tokens = indexes_to_tokens($comp->{pe}, $token_index->{pe});
        @mt_tokens = grep {is_token_content($_)} @mt_tokens;
        @pe_tokens = grep {is_token_content($_)} @pe_tokens;
        !tokens_equal(\@mt_tokens, \@pe_tokens) && 
        (join "", @mt_tokens) ne (join "", @pe_tokens)
    } @new_ali_comps;
    #print STDERR "ALI_COMPS3: ".scalar(@new_ali_comps)."\n";
    
    return \@new_ali_comps;
}

sub indexes_to_tokens {
    my ($token_idxs, $tokens) = @_;
    return if (!defined $token_idxs);
    return map {$tokens->[$_]} @$token_idxs;
}

sub print_components {
    my ($fh, $ali_comps, $token_index) = @_;
    foreach my $comp (@$ali_comps) {
        my @phrases = ();
        foreach my $side_name (@SIDES_ORDER) {
            my $token_idxs = $comp->{$side_name};
            my @tokens = ();
            if (defined $token_idxs) {
                @tokens = map {$token_index->{$side_name}[$_]} @$token_idxs;
            }
            push @phrases, (join " ", @tokens);
        }
        print {$fh} join "\t", @phrases;
        print {$fh} "\n";
    }
    print {$fh} "\n";
}

sub print_annot_for_tokens {
    my ($ali_comps, $token_index) = @_;
    my %complex_src_indexes = map {my $comp = $_; map {$_ => 1} @{$comp->{src}}} @$ali_comps;
    my @annots = ();
    print STDERR scalar @{$token_index->{src}}."\n";
    for (my $i = 0; $i < scalar(@{$token_index->{src}}); $i++) {
        my $annot = $complex_src_indexes{$i} ? 1 : 0;
        push @annots, $annot;
        # push @annots, $token_index->{src}[$i].":".$annot;
    }
    print join " ", @annots;
    print "\n";
}


open my $src_mt_bitext_f, "<".($ARGV[0] =~ /\.gz$/ ? ":gzip" : "").":utf8", $ARGV[0];
open my $src_pe_bitext_f, "<".($ARGV[1] =~ /\.gz$/ ? ":gzip" : "").":utf8", $ARGV[1];
open my $src_mt_align_f, "<".($ARGV[2] =~ /\.gz$/ ? ":gzip" : "").":utf8", $ARGV[2];
open my $src_pe_align_f, "<".($ARGV[3] =~ /\.gz$/ ? ":gzip" : "").":utf8", $ARGV[3];
open my $pe_mt_align_f, "<".($ARGV[4] =~ /\.gz$/ ? ":gzip" : "").":utf8", $ARGV[4];

my $out_components_fh;
if (defined $out_components_path) {
    open $out_components_fh, ">".($out_components_path =~ /\.gz$/ ? ":gzip" : "").":utf8", $out_components_path;
}

my $line = 0;
while (my $src_mt_bitext_l = <$src_mt_bitext_f>) {
    my $src_pe_bitext_l = <$src_pe_bitext_f>;
    my $src_mt_align_l = <$src_mt_align_f>;
    my $src_pe_align_l = <$src_pe_align_f>;
    my $pe_mt_align_l = <$pe_mt_align_f>;
    my ($ali_comps, $token_index) = extract_from_sent_triple($src_mt_bitext_l, $src_pe_bitext_l, $src_mt_align_l, $src_pe_align_l, $pe_mt_align_l);
    $ali_comps = filter_ali_comps($ali_comps, $token_index);
    print_annot_for_tokens($ali_comps, $token_index);
    if (defined $out_components_fh) {
        print_components($out_components_fh, $ali_comps, $token_index);
    }
    $line++;
    #last if ($line == 2276);
}

close $src_mt_bitext_f;
close $src_pe_bitext_f;
close $src_mt_align_f;
close $src_pe_align_f;
close $pe_mt_align_f;

if (defined $out_components_fh) {
    close $out_components_fh;
}
