from os.path import join

base = '< BASE PATH FOR PROJECT HERE >'

dat = join(base, 'dat')

# contentwise paths:

cw_base = '< PATH FOR CONTENTWISE DATA HERE >'

# obtain contentwise data (https://dl.acm.org/doi/abs/10.1145/3340531.3412774) and put them under cw_base folder:
cw_dat_paths = {'positive_impressions': join(cw_base, 'impressions-direct-link.csv.gz'),
                'negative_impressions': join(cw_base, 'impressions-non-direct-link.csv.gz'),
                'interactions': join(cw_base, 'interactions.csv.gz')}

cw_stages = {'intermediate_0': join(cw_base,'processed-0.csv.gz'),
             'intermediate_1': join(cw_base, 'processed-state-subitems.csv.gz'), 
             'output': {'train': join(cw_base, 'ContentWise-train-subitems.csv.gz'),
                        'validate': join(cw_base, 'ContentWise-validate-subitems.csv.gz'),
                        'test-seen': join(cw_base, 'ContentWise-test-seen-users-subitems.csv.gz'),
                        'test-unseen': join(cw_base, 'ContentWise-test-unseen-users-subitems.csv.gz'),
                        'alldat': join(cw_base, 'ContentWise-alldat-subitems.csv.gz'),
                        'hypsubset': join(cw_base, 'ContentWise-hypsubset-subitems.csv.gz')}
            }

# file path/stem for storing data for training NMF models used in temperature sweep experiment:
nmf_dat_stem = join(dat, 'rec_model_nmf_dat_%i.h5')
visit_model = dat