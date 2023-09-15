import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import uproot

# SM input features
input_features = ["nrecojet_antikt4","nrecojet_antikt4_btag85","chi_hh","chi_hz","chi_zz","DeltaR_12", "DeltaR_34", "DeltaPhi_12", "DeltaPhi_34", "DeltaEta_12", "DeltaEta_34", "DeltaR_1234", "DeltaPhi_1234", "DeltaEta_1234", "DeltaEta_max", "DeltaEta_min", "DeltaEta_mean", "DeltaR_max", "DeltaR_min", "DeltaR_mean", "mH1_hh", "mH2_hh", "mH_hz", "mZ_hz", "mZ1_zz", "mZ2_zz","pT_1", 'pT_2', "pT_3", "pT_4", "eta_1", "eta_2", "eta_3", "eta_4", "pT_12", "pT_34", "sumHT_bjets", "sumHT_totalJets", "met_pt", "met_phi"]
labels = [r"nrecojet_antikt4",r"nrecojet_antikt4_btag85",r"chi_hh",r"chi_hz",r"chi_zz",r"DeltaR_12", r"DeltaR_34", r"DeltaPhi_12", r"DeltaPhi_34", r"DeltaEta_12", r"DeltaEta_34", r"DeltaR_1234", r"DeltaPhi_1234", r"DeltaEta_1234", r"DeltaEta_max", r"DeltaEta_min", r"DeltaEta_mean", r"DeltaR_max", r"DeltaR_min", r"DeltaR_mean", r"mH1_hh", r"mH2_hh", r"mH_hz", r"mZ_hz", r"mZ1_zz", r"mZ2_zz",r"pT_1", r'pT_2', r"pT_3", r"pT_4", r"eta_1", r"eta_2", r"eta_3", r"eta_4", r"pT_12", r"pT_34", r"sumHT_bjets", r"sumHT_totalJets", r"met_pt", r"met_phi"]
bkg_file = uproot.open("/eos/user/g/gstucchi/NTRUPLES/FullHadCuts/background.root:MiniTree_NOSYS")
df_bkg = bkg_file.arrays(library="pd").loc[:, input_features]
bkg_corr = df_bkg.corr()
mask = numpy.triu(numpy.ones_like(bkg_corr, dtype=bool))
f, ax = plt.subplots(figsize=(10, 8))
mat = sns.heatmap(abs(bkg_corr),
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, xticklabels=labels, yticklabels=labels)
colorbar = mat.collections[0].colorbar
colorbar.ax.set_ylabel('Absolute correlation', rotation=270, labelpad=15, fontsize=15)
f.tight_layout()
f.savefig("bkg_SM_nn_corr.pdf")


sig_file = uproot.open("/eos/user/g/gstucchi/NTRUPLES/FullHadCuts/signal.root:MiniTree_NOSYS")
df_sig = sig_file.arrays(library="pd").loc[:, input_features]
sig_corr = df_sig.corr()
mask = numpy.triu(numpy.ones_like(sig_corr, dtype=bool))
f, ax = plt.subplots(figsize=(10, 8))
mat = sns.heatmap(abs(sig_corr),
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, xticklabels=labels, yticklabels=labels)
colorbar = mat.collections[0].colorbar
colorbar.ax.set_ylabel('Absolute correlation', rotation=270, labelpad=15, fontsize=15)
f.tight_layout()
f.savefig("sig_SM_nn_corr.pdf")
