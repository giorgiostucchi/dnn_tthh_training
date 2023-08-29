import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import uproot

# Resonant TRSM input features
#input_features = ["rmsmBB","rmsdABB","aplan4dv2b","massfraceta","htbjet","mHcosTheta","sphere3dbtrans","bjetH1_dR","bjetH2_dR","bjetH3_dR"]
#labels = [r"rms $m_H$", r"rms $\Delta A_H$", r"Aplanarity$_{6 jets}$",r"$\eta$-$m_{HHH}$ fraction", r"$H_{T_{6 jets}}$", r"$m_H cos\theta$", r"Transverse sphericity$_{6 jets}$",r"$\Delta R_{H1}$",
#          r"$\Delta R_{H2}$", r"$\Delta R_{H3}$"]

# SM input features
input_features = ["mH1","mH2","bjetH1_dR","bjetH2_dR","rmsmH","rmsEta","mHcosTheta","sphere3dbtrans","aplan3dv2b","rmsdRBB"]
labels = [r"$m_H1$",r"$m_H2$",r"$\Delta R_{H1}$",r"$\Delta R_{H2}$",r"rms $m_H$",r"rms $\eta$",r"$m_H cos\theta$",r"Trans. sphericity$_{6 jets}$",r"Aplanarity$_{6 jets}$",r"rms $\Delta R_H$"]
even_bkg_file = uproot.open("/data/atlas/atlasdata3/maggiechen/HHH6b_pairing/pairing+nn/20230501_pTreco_off/SM_HHH/training_inclusive/5b_data_even.root:tree")
df_bkg_even = even_bkg_file.arrays(library="pd").loc[:, input_features]
bkg_corr = df_bkg_even.corr()
mask = numpy.triu(numpy.ones_like(bkg_corr, dtype=bool))
f, ax = plt.subplots(figsize=(10, 8))
mat = sns.heatmap(abs(bkg_corr),
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, xticklabels=labels, yticklabels=labels)
colorbar = mat.collections[0].colorbar
colorbar.ax.set_ylabel('Absolute correlation', rotation=270, labelpad=15, fontsize=15)
f.tight_layout()
f.savefig("5b_data_bkg_SM_nn_corr.pdf")

#even_sig_file = uproot.open("/data/atlas/atlasdata3/maggiechen/HHH6b_pairing/pairing+nn/20230501_pTreco_off/mH-120_115_110_pairing/training_inclusive/6b_res_TRSM_even.root:tree")
even_sig_file = uproot.open("/data/atlas/atlasdata3/maggiechen/HHH6b_pairing/pairing+nn/20230501_pTreco_off/SM_HHH/training_inclusive/6b_SM_HHH_521162_even.root:tree")
df_sig_even = even_sig_file.arrays(library="pd").loc[:, input_features]
sig_corr = df_sig_even.corr()
mask = numpy.triu(numpy.ones_like(sig_corr, dtype=bool))
f, ax = plt.subplots(figsize=(10, 8))
mat = sns.heatmap(abs(sig_corr),
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, xticklabels=labels, yticklabels=labels)
colorbar = mat.collections[0].colorbar
colorbar.ax.set_ylabel('Absolute correlation', rotation=270, labelpad=15, fontsize=15)
f.tight_layout()
f.savefig("6b_sig_10_SM_nn_corr.pdf")