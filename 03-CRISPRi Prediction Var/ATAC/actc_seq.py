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

def ATACseq_run(actual_tiling , bigbed_df, sgrna_location_col, chromosome_col = 'chromosome', gene_col = 'gene symbol'):
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
        print(chromo)
        chrom_num_df = peak_overlap(actual_tiling = actual_tiling, bigbed_df = bigbed_df, chrom = chromo, 
                                    sgrna_location_col = sgrna_location_col, 
                                    chromosome_col = chromosome_col, gene_col = gene_col)
        anno_peak = pd.concat([anno_peak, chrom_num_df])
        
    return anno_peak


# pval and graphs
def pvalue_overlap_comparison_boxplot(df, phenotype, gene_col = 'gene symbol', 
                                      test_direction = 'no peak < peak', 
                                      binary_col = 'overlap with peak'):
    # This function produce pvalues and boxplots comparsion of peak vs no peak 
    # df: is the outputs from the function ATACseq_run
    # phenotype: is the name of the column that has phenotyic values
    # gene_col: is the name of the column that has gene symbol
    # test_direction: {'no peak < peak', 'no peak > peak', 'two-sided'} 
    # string that indicates the direction of mannwhitneyu or defines the alternative hypothesis
    # 'no peak < peak': the distribution underlying no peak is stochastically less than the distribution underlying peak
    # 'no peak > peak': the distribution underlying no peak is stochastically greater than the distribution underlying peak
    # 'two-sided': the distributions are not equal
    # more info can be found https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    
    #returns a list of pvalues for all qualified genes and box plots of peak vs no peak comparsion
    fig, axes =plt.subplots(math.ceil(len(df[gene_col].unique())/5),5, 
                            figsize=(20,math.ceil(len(df[gene_col].unique())/5)*3.5), sharex=False)
    axes = axes.flatten()
    #pvals_list = []
    pvals_dict = {}

    if test_direction == 'no peak < peak':
        alter = 'less'
    if test_direction == 'no peak > peak':
        alter = 'greater'
    if test_direction == 'two-sided':
        alter = 'two-sided'
        
    for ax, gene in zip(axes, df[gene_col].unique()):
        gene_sp = df[df[gene_col].isin([gene])]
        if (sum(gene_sp[binary_col] == 1) >= 10) & (sum(gene_sp[binary_col] == 0) >= 10):
            sns.boxplot(data=gene_sp
            , x=binary_col ,y = phenotype, order=[0,1], 
                        ax = ax, palette=['grey','orange'], legend=False, hue=binary_col).set(title=gene)
            ax.set_ylabel(phenotype) 
            ax.set_xlabel("")    
            sns.despine()
            _, pval = mannwhitneyu(gene_sp[gene_sp[binary_col] == 0][phenotype], 
                               gene_sp[gene_sp[binary_col] == 1][phenotype], 
                               alternative=alter, method="asymptotic")
            #pvals_list.append(pval)
            pvals_dict[gene] = pval
            plt.text(.02, 0.92, 'P: {:.4f}'.format(pval), transform=ax.transAxes)
            plt.xticks([0,1], 
                       [ "no peak, n="+str(sum(gene_sp[binary_col] == 0)) , "has peak, n="+str(sum(gene_sp[binary_col] == 1))])
        else:
            print(gene+': Sample size in overlap or nonoverlap < 10')
    return pvals_dict#pvals_list

def pvalue_overlap_comparison_boxplot_nontiling(df, phenotype, cell_line, gene_col = 'gene_type', 
                                      test_direction = 'no peak < peak', 
                                      binary_col = 'overlap with peak'):
     # This function produce pvalues and boxplots comparsion of peak vs no peak but ONLY for non-tiling
    # df: is the outputs from the function ATACseq_run
    # phenotype: is the name of the column that has phenotyic values, normalized!!!
    # gene_col: is the name of the column that has gene type
    # test_direction: {'no peak < peak', 'no peak > peak', 'two-sided'} 
    # string that indicates the direction of mannwhitneyu or defines the alternative hypothesis
    # 'no peak < peak': the distribution underlying no peak is stochastically less than the distribution underlying peak
    # 'no peak > peak': the distribution underlying no peak is stochastically greater than the distribution underlying peak
    # 'two-sided': the distributions are not equal
    # more info can be found https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    if test_direction == 'no peak < peak':
        alter = 'less'
    if test_direction == 'no peak > peak':
        alter = 'greater'
    if test_direction == 'two-sided':
        alter = 'two-sided'
        
    ax = sns.boxplot(data=df, x=binary_col, y=phenotype, hue =  gene_col) 
    
    plt.xticks([0,1], 
                   [ "no peak, n="+str(sum(df[binary_col] == 0)) , 
                    "has peak, n="+str(sum(df[binary_col] == 1))])  
    pval_dic = {}
    y = 0.92
    for num_hum in df[gene_col].unique():
        spc_df = df[df[gene_col] == num_hum]
        _, pval = mannwhitneyu(spc_df[spc_df[binary_col] == 0][phenotype], 
                                   spc_df[spc_df[binary_col] == 1][phenotype], 
                                   alternative=alter, method="asymptotic")
        pval_dic[num_hum] = pval
        plt.text(1.05, y, num_hum+' pval: {:.3f}'.format(pval), transform=ax.transAxes)
        y = y- 0.07# 0.85
        
    ax.set(title='Comparsion of ATAC seq overlap based on cell '+ cell_line)
    ax.legend(loc='center left', title = 'Gene type',bbox_to_anchor=(1, 0.5))
    return pval_dic

    
def scatterplot_overlap(df, phenotype, distance_to_tss, gene_col = 'gene symbol', 
                                      binary_col = 'overlap with peak'):
    # This function produce pvalues and boxplots comparsion of peak vs no peak 
    # df: is the outputs from the function ATACseq_run
    # distance_to_tss: is the name of the column for distance to tss
    # phenotype: is the name of the column that has phenotyic values
    # gene_col: is the name of the column that has gene symbol
    # binary_col: is the name of the column that indicates overlap vs no overlap with peak
    
    #returns scatterplots where each plot is a gene and each point is a guide. 
    # Orange guide means they overlapped with a peak, otherwise they are not. 
    fig, axes =plt.subplots(math.ceil(len(df[gene_col].unique())/5),5, 
                            figsize=(20,math.ceil(len(df[gene_col].unique())/5)*3.5), sharex=False)
    axes = axes.flatten()

    genes_w_overlap = df[df['overlap with peak'] == 1][gene_col].unique()
    genes_w_overlap_df = df[df[gene_col].isin(genes_w_overlap)]

    for ax, gene in zip(axes, genes_w_overlap):
        sns.scatterplot(data= genes_w_overlap_df[(genes_w_overlap_df[gene_col].isin([gene])) &

                                     (genes_w_overlap_df[binary_col] == 1)], 
                        x=sgrna_location_col, 
                        y=phenotype, ax = ax, color = 'orange')
        sns.scatterplot(data= genes_w_overlap_df[(genes_w_overlap_df[gene_col].isin([gene])) &

                                     (genes_w_overlap_df[binary_col] == 0)], 
                        x=sgrna_location_col, 
                        y=phenotype, ax= ax, color = 'grey', alpha = 0.2).set(title=gene)

        ax.set_ylabel(phenotype) 
        sns.despine()

        
def pval_agg(pval_list, title):
    #This function produce a histgram that visually show aggregate pvalues from the function[pvalue_overlap_comparison_boxplot]
    # pval_list: outputs from pvalue_overlap_comparison_boxplot
    # title: title of the histgram
    fig, ax = plt.subplots()
    sns.histplot(pval_list, bins = 20).set(title = title)
    ax.set_ylabel("Count of Genes") 
    ax.set_xlabel("P-value")

    
def scatterplot_overlap_distance_tss(df, phenotype, distance_to_tss, gene_col = 'gene symbol', 
                                      binary_col = 'overlap with peak'):
    # This function produce pvalues and boxplots comparsion of peak vs no peak 
    # df: is the outputs from the function ATACseq_run
    # distance_to_tss: is the name of the column for distance to tss
    # phenotype: is the name of the column that has phenotyic values
    # gene_col: is the name of the column that has gene symbol
    # binary_col: is the name of the column that indicates overlap vs no overlap with peak
    
    #returns scatterplots where each plot is a gene and each point is a guide. 
    # Orange guide means they overlapped with a peak, otherwise they are not. 
    fig, axes =plt.subplots(math.ceil(len(df[gene_col].unique())/5),5, 
                            figsize=(20,math.ceil(len(df[gene_col].unique())/5)*3.5), sharex=False)
    axes = axes.flatten()

    genes_w_overlap = df[df['overlap with peak'] == 1][gene_col].unique()
    genes_w_overlap_df = df[df[gene_col].isin(genes_w_overlap)]

    for ax, gene in zip(axes, genes_w_overlap):
        sns.scatterplot(data= genes_w_overlap_df[(genes_w_overlap_df[gene_col].isin([gene])) &

                                     (genes_w_overlap_df[binary_col] == 1)], 
                        x=distance_to_tss, 
                        y=phenotype, ax = ax, color = 'orange')
        sns.scatterplot(data= genes_w_overlap_df[(genes_w_overlap_df[gene_col].isin([gene])) &

                                     (genes_w_overlap_df[binary_col] == 0)], 
                        x=distance_to_tss, 
                        y=phenotype, ax= ax, color = 'grey', alpha = 0.2).set(title=gene)

        ax.set_ylabel(phenotype) 
        sns.despine()

        
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



