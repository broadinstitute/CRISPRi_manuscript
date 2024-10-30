import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statannot import add_stat_annotation
from scipy.stats import mannwhitneyu
import pyBigWig
from tqdm import tqdm
import math


def isInRange(n, tbl):
    #https://stackoverflow.com/questions/9019581/what-does-numpy-apply-along-axis-perform-exactly
    #https://stackoverflow.com/questions/71377816/check-if-value-is-between-two-values-in-numpy-array
    # n: is sgRNA location in the tiling/phenotypic dataset
    # tbl: numpy file of dataframe of bigbed with chro, start, and end of peak filtered_peak_file.to_numpy()
    #apply_along_axis applies the supplied function along 1D slices of the input array, 
    #with the slices taken along the axis you specify. 
    return sum(np.apply_along_axis(lambda row: row[0] <= n <= row[1], 1, tbl)\
        .tolist())

def peak_overlap(actual_tiling , bigbed_df, chrom, sgrna_location_col, chromosome_col = 'chromosome', 
                 gene_col = 'gene symbol'):
    # This function find whether guide overlaps with a peak in one chromosome 
    
    # actual_tiling: CRISPRi data (or any CRISPR data) with at least three columns 
    #                       [a location to indicate sgRNA, Gene name, Chromosome]
    #                       Note this DOES NOT require unique sgRNA location 
    # bigbed_df: ATAC seq from ENCODE in bigBed format and read through pyBigWig
    # chrom: string of chromosome such as 'chr1'
    # sgrna_location_col: string of column for sgRNA location
    
    # returns a dataframe, same as the actual_tilling but with an additional column overlap with peak to indicate 
    # whether theres an overlap between start and end of a peak and sgRNA location for one chromosome
    
    #----------------------------------------------------------------------------------------------------
    print(chrom)
    #subset chromsome number
    tiling_subset_chromo =  actual_tiling[actual_tiling[chromosome_col] == chrom].copy(deep=True)
    
    # select unique pam coord and chr and gene - remove duplicate 
    # tiling_lib: CRISPRi data (or any CRISPR data) with at least three columns 
    #                       [a location to indicate sgRNA, Gene name, Chromosome]
    #                       Note this should has unique sgRNA location 
    tiling_lib = tiling_subset_chromo.drop_duplicates(subset=[sgrna_location_col, chromosome_col,gene_col]).copy(deep=True)
    
    
    # change from string to int
    tiling_lib[sgrna_location_col] = list(map(int,tiling_lib[sgrna_location_col]))
    
    #obtain the smallest pam coord in a specific chromosome
    smallest_pam_coord = tiling_lib[sgrna_location_col].min()
    #obtain the largest pam coord in a specific chromosome
    largestest_pam_coord = tiling_lib[sgrna_location_col].max()

    
    #subset chrom number and having the end coord to be larger than the smallest pam coord
    
    # Retrieving bigBed entries in https://github.com/deeptools/pyBigWig explains
    # filtered_peak_file returns a list of tuple of (Start position in chromosome, End position in chromosome)
    filtered_peak_file = bigbed_df.entries(chrom, smallest_pam_coord, largestest_pam_coord, withString=False) 
    
    new_tiling_lib = tiling_lib.copy(deep=True)

    # make sure the selected peak file is not none
    if filtered_peak_file is not None:
    
        # only kept unique peaks and make it into dataframe because of https://www.biostars.org/p/464618/
        filtered_peak_file = pd.DataFrame(set(filtered_peak_file))


        # iterating over every single sgRNA location in the tiling library/a dataset with phenotypic data
        peak_list = [isInRange(x, filtered_peak_file.to_numpy()) 
                     for x in tqdm(np.nditer(tiling_lib[sgrna_location_col].to_numpy()), 
                                   total=len(tiling_lib[sgrna_location_col]),
                                   desc='number of unique sgRNA position and gene symbol for ' + chrom)]
    # if its None then return a list of 0 to show there is no overlaps
    else:
        peak_list = [0] *len(new_tiling_lib)

    
    #https://stackoverflow.com/questions/32573452/settingwithcopywarning-even-when-using-locrow-indexer-col-indexer-value
    new_tiling_lib.loc[:, 'overlap with peak'] = peak_list
    
    new_tiling_lib = new_tiling_lib[[gene_col, chromosome_col, sgrna_location_col, 'overlap with peak']]
    #b['overlap with peak'] = peak_list
    chr_df = pd.merge(actual_tiling, new_tiling_lib, on = [gene_col, chromosome_col, sgrna_location_col])

    return chr_df


        
######## DHS----------------

def dhs_peak_overlap(actual_tiling , bigbed_df, chrom, sgrna_location_col, chromosome_col = 'chromosome', 
                 gene_col = 'gene symbol'):
    # This function find whether guide overlaps with a peak in one chromosome 
    
    # actual_tiling: CRISPRi data (or any CRISPR data) with at least three columns 
    #                       [a location to indicate sgRNA, Gene name, Chromosome]
    #                       Note this DOES NOT require unique sgRNA location 
    # bigbed_df: ATAC seq from ENCODE in bigBed format and read through pyBigWig
    # chrom: string of chromosome such as 'chr1'
    # sgrna_location_col: string of column for sgRNA location
    
    # returns a dataframe, same as the actual_tilling but with an additional column overlap with peak to indicate 
    # whether theres an overlap between start and end of a peak and sgRNA location for one chromosome
    
    #----------------------------------------------------------------------------------------------------
        
    #subset chromsome number
    tiling_subset_chromo =  actual_tiling[actual_tiling[chromosome_col] == chrom].copy()
    
    # select unique pam coord and chr and gene - remove duplicate 
    # tiling_lib: CRISPRi data (or any CRISPR data) with at least three columns 
    #                       [a location to indicate sgRNA, Gene name, Chromosome]
    #                       Note this should has unique sgRNA location 
    tiling_lib = tiling_subset_chromo.drop_duplicates(subset=[sgrna_location_col, chromosome_col,gene_col]).copy()
    
    
    # change from string to int
    tiling_lib[sgrna_location_col] = list(map(int,tiling_lib[sgrna_location_col]))
    
    #obtain the smallest pam coord in a specific chromosome
    smallest_pam_coord = tiling_lib[sgrna_location_col].min()
    #obtain the largest pam coord in a specific chromosome
    largestest_pam_coord = tiling_lib[sgrna_location_col].max()

    
    #subset chrom number and having the end coord to be larger than the smallest pam coord
    
    # Retrieving bigBed entries in https://github.com/deeptools/pyBigWig explains
    # filtered_peak_file returns a list of tuple of (Start position in chromosome, End position in chromosome)
    # filtered_peak_file = bigbed_df.entries(chrom, smallest_pam_coord, largestest_pam_coord, withString=False) 
    filtered_peak_file = bigbed_df[(bigbed_df["start"] >= int(smallest_pam_coord)) & (bigbed_df["end"] <= int(largestest_pam_coord))].copy()
    # only kept unique peaks and make it into dataframe because of https://www.biostars.org/p/464618/
    # filtered_peak_file = pd.DataFrame(set(filtered_peak_file))
    filtered_peak_file = filtered_peak_file[["start", "end"]].drop_duplicates()
    
       
    # iterating over every single sgRNA location in the tiling library/a dataset with phenotypic data
    peak_list = [isInRange(x, filtered_peak_file.to_numpy()) 
                 for x in tqdm(np.nditer(tiling_lib[sgrna_location_col].to_numpy()), 
                               total=len(tiling_lib[sgrna_location_col]),
                               desc='number of unique sgRNA position and gene symbol for ' + chrom)]


    new_tiling_lib = tiling_lib.copy()
    
    #https://stackoverflow.com/questions/32573452/settingwithcopywarning-even-when-using-locrow-indexer-col-indexer-value
    new_tiling_lib.loc[:, 'overlap with peak'] = peak_list
    
    new_tiling_lib = new_tiling_lib[[gene_col, chromosome_col, sgrna_location_col, 'overlap with peak']]
    #b['overlap with peak'] = peak_list
    chr_df = pd.merge(actual_tiling, new_tiling_lib, on = [gene_col, chromosome_col, sgrna_location_col])

    return chr_df

def DHS_run(actual_tiling , bigbed_df, sgrna_location_col, chromosome_col = 'chromosome', gene_col = 'gene symbol'):
    anno_peak = pd.DataFrame()
    
    # This function iterate over all the chromosome to run the function [peak_overlap] and concatenate them together
    # actual_tiling: CRISPRi data (or any CRISPR data) with at least three columns 
    #                       [a location to indicate sgRNA, Gene name, Chromosome]
    #                       Note this DOES NOT require unique sgRNA location 
    # bigbed_df: ATAC seq from ENCODE in bigBed format and read through pyBigWig
    # sgrna_location_col: string of column name for sgRNA location
    # chromosome_col: string of column name for chromosome
    # gene_col: string of column name for 'gene symbol'
    
    # returns a dataframe, same as the actual_tilling but with an additional column overlap with peak to indicate 
    # whether theres an overlap between start and end of a peak and sgRNA location for mutiple chromosomes
    
    # a list of chromosome in the data
    chromo_list = set(actual_tiling[chromosome_col])
    print('There are total', len(chromo_list), 
      ' chromosome in the dataset and they are ',set(chromo_list)) 
    
    #iterate peak_overlap function for all the chromosome
    for chromo in chromo_list:
        chrom_num_df = dhs_peak_overlap(actual_tiling = actual_tiling, bigbed_df = bigbed_df, chrom = chromo, 
                                    sgrna_location_col = sgrna_location_col, 
                                    chromosome_col = chromosome_col, gene_col = gene_col)
        anno_peak = pd.concat([anno_peak, chrom_num_df])
        
    return anno_peak



