import os
import sys
from urllib.request import HTTPError, urlretrieve
from zipfile import ZipFile

import fastobo
import networkx as nx
import numpy as np
import pandas as pd


# Variables
data_path = './go_hagr/'
hagr_url = 'https://www.genomics.senescence.info/genes/models_genes.zip'
hagr_path = data_path + 'hagr/'
hagr_filename = 'hagr.zip'
hagr_full_path = hagr_path + hagr_filename
go_url = 'https://purl.obolibrary.org/obo/go.obo'
# go_url = 'https://zenodo.org/records/10536401/files/go-release-archive.tgz' # TODO consider this in the future
go_path = data_path + 'go/'
go_filename = 'go.obo'
go_full_path = go_path + go_filename
gaf_path = data_path + 'gaf/'
dataset_path = data_path + 'datasets/'


download_url = {
    'Caenorhabditis elegans':
        'https://current.geneontology.org/annotations/wb.gaf.gz',
    'Mus musculus':
        'https://current.geneontology.org/annotations/mgi.gaf.gz',
    'Saccharomyces cerevisiae':
         'https://current.geneontology.org/annotations/sgd.gaf.gz',
    'Drosophila melanogaster':
        'https://current.geneontology.org/annotations/fb.gaf.gz'
}

# TODO Should be replaced by the actual count of rows with an exclamation mark '!' in the respective files
start_row = {
    'Caenorhabditis elegans': 35,
    'Mus musculus': 36,
    'Saccharomyces cerevisiae': 35,
    'Drosophila melanogaster': 33
}


# Functions
def gaf_filename(species):
    return str.lower(species).replace(" ", "_") + ".gaf.gz"

def gaf_full_path(gaf_path, species):
    return gaf_path + gaf_filename(species)

def mask(
        df : pd.DataFrame,
        col : int,
        term : str,
        strict : bool
):
    if strict:
        t = '^'    
    else:
        t = ''
    return df[col].str.contains((t + term + '$'), regex=True, na=False)


# --------------------------------------------------------------------------------
#                   Beginning of the actual code
# --------------------------------------------------------------------------------

# Create folders for respective data/files
for path in [data_path, hagr_path, go_path, gaf_path, dataset_path]:
    try:
        os.makedirs(path)
        print("Folder %s created." % path)
    except FileExistsError:
        print("Folder %s already exists." % path)

# # Download the relevant HAGR and GO files
# for dataset, url, full_path in [ ('HAGR', hagr_url, hagr_full_path) , ('GO', go_url, go_full_path) ]:
#     print(dataset, url, full_path)
#     try:
#         urlretrieve(url, hagr_full_path)
#         print("%s dataset downloaded." % dataset)
#     except HTTPError:
#         print("%s download failed." % dataset)
#         sys.exit()

# Unzip and read the relevant file from HAGR
zf = ZipFile(hagr_full_path)
hagr = pd.read_csv(zf.open('genage_models.csv'))

# Extract all species from HAGR
hagr_species = list(hagr.organism.unique())

# Read relevant files from GO
go = fastobo.load(go_full_path)
go_dag = nx.DiGraph()

# Construct networkx digraph from GO terms and their IS_A relationships
for go_term in go:
    if isinstance(go_term, fastobo.term.TermFrame):
        for clause in go_term:
            if isinstance(clause, fastobo.term.NamespaceClause):
                go_dag.add_node(str(go_term.id), namespace=clause.raw_value())
            if isinstance(clause, fastobo.term.IsAClause):
                go_dag.add_edge(str(go_term.id), str(clause.term))

# Save GO DAG as GML file
nx.write_gml(G=go_dag, path=(dataset_path + 'go_dag.gml'))

# Per species: Download and process Gene Annotation Files (GAF) to assign GO terms to genes
for species in hagr_species:
    if species not in download_url.keys():
        pass
    else:
        gaf_species_full_path = gaf_full_path(gaf_path, species)

        try:
            urlretrieve(download_url[species], gaf_species_full_path)
            print("GAF data for %s downloaded." % str.upper(species))
        # TODO put entire code below here
        except HTTPError:
            print("GAF download for %s failed." % str.upper(species))
        except ValueError:
            print("[INFO] %s is not in URL dictionary" % str.upper(species))

        # Read the GAF files
        # TODO Calculate the number of skiprows from files, example follows (but requires prior unpacking)
        # num_lines = os.popen("grep -ic \"\!\" \"./wb.gaf\"").read().split("\n")[0]
        gaf = pd.read_csv(gaf_species_full_path, compression="gzip", skiprows=start_row[species], header=None, sep="\t", quotechar='"', low_memory=False)

        # Extract unique gene symbols per species
        hagr_genes = list(hagr.loc[((hagr["organism"]==species) & (hagr["longevity influence"].str.contains('Pro-Longevity|Anti-Longevity'))), "symbol"].unique())

        # Retrieve GO terms as dict from annotation file
        count3 = 0
        count10 = 0
        count0 = 0

        count3_ = 0
        count10_ = 0
        count0_ = 0

        gene_annotations = {}

        not_mask = ~gaf[3].str.contains(('^NOT'), regex=True, na=False)

        for gene in hagr_genes:
            gene_mask = mask(df=gaf, col=2, term=gene, strict=True)
            if (gene_mask & not_mask).any():
                go_list = list(gaf.loc[(gene_mask & not_mask), 4])
                count3+=1
            else:
                synonym_mask = mask(df=gaf, col=10, term=gene, strict=False)
                if (synonym_mask & not_mask).any():
                    go_list = list(gaf.loc[(synonym_mask & not_mask), 4])
                    count10+=1
                else:
                    try:
                        gene_part = gene.split('_')[1]
                        # print("After split", gene)
                        gene_mask = mask(df=gaf, col=2, term=gene_part, strict=True)
                        if (gene_mask & not_mask).any():
                            go_list = list(gaf.loc[(gene_mask & not_mask), 4])
                            count3_+=1
                        else:
                            synonym_mask = mask(df=gaf, col=10, term=gene_part, strict=False)
                            if (synonym_mask & not_mask).any():
                                go_list = list(gaf.loc[(synonym_mask & not_mask), 4])
                                count10_+=1
                            else:
                                count0_+=1
                                go_list = []
                    except IndexError:
                        # print("Index error", gene)
                        go_list = []
                        count0+=1
            if go_list: gene_annotations[gene] = go_list

        print("c3", count3)
        print("c10", count10)
        print("c0", count0)

        print("c3_", count3_)
        print("c10_", count10_)
        print("c0_", count0_)

        # Convert dict into dataset (numpy array: rows = genes, columns = dataset_go_terms)
        num_rows = len(gene_annotations.keys()) # TODO Check if correct
        dataset_columns = []
        go_annotations = {}

        for gene in gene_annotations:
            row_index = list(gene_annotations.keys()).index(gene)
            for item in gene_annotations[gene]:
                column_values = np.full((num_rows, 1), fill_value=False)
                column_values[row_index] = True
                if item not in dataset_columns:
                    dataset_columns.append(item)
                else:
                    column_index = dataset_columns.index(item)
                    column_values = np.logical_or(go_annotations[item], column_values)
                go_annotations[item] = column_values

        dataset = np.concatenate([x for x in go_annotations.values()], axis = 1)

        # Extract empty rows and delete them from the dataset
        empty_rows = []
        for row in range(dataset.shape[0]):
            if dataset[row,:].any() == False:
                empty_rows.append(row)
        dataset = np.delete(dataset, empty_rows, axis=0)

        # Row description list
        dataset_rows = [gene for gene_index, gene in enumerate(gene_annotations.keys()) if gene_index not in empty_rows]

        # Generate list of target variables
        target_variables = []
        for annotatable_gene in gene_annotations.keys():
            if hagr.loc[(hagr["symbol"]==annotatable_gene) & (hagr['organism']==species), ['longevity influence']].to_numpy()[0] == 'Pro-Longevity':
                target_variables.append(1)
            else:
                target_variables.append(0)

        # Save dataset as numpy arrays
        dataset_full_path = dataset_path + 'dataset_' + species.split('_')[0] + '_' + species.split('_')[1] + '.npz'
        np.savez(file=dataset_full_path, dataset=dataset, dataset_rows=dataset_rows, dataset_columns=dataset_columns, target_variables=target_variables)
        