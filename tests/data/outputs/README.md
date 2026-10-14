# Test outputs

A minimal set of ONCVPSP/METAPSP outputs which together exercise every feature seen in a
collection of ~96,500 runs: program versions and relativistic treatments, each error and
warning, input options, and optional output sections. Each file is listed with the
features that no other file in the set covers.

| File | Source | Features unique to this file |
|---|---|---|
| `metapsp-1.0.1-nr_20_Ca2_m.out.gz` | code/metapsp/tests/refs/20_Ca2_m.out | `lpopt:1`, `psfile:psml`, `psml`, `warn:localized_bound_state_for_projector` |
| `metapsp-1.0.1-nr_74_W_d1_m.out.gz` | code/metapsp/tests/refs/74_W_d1_m.out | `nproj_combo:(3,)`, `psfile:none` |
| `oncvpsp-2.1.1-fr_08_O_r.out.gz` | code/pseudo_dojo/pseudo_dojo/ppcodes/tests/08_O_r.out | `prog:ONCVPSP-2.1.1:fully-relativistic` |
| `oncvpsp-2.1.1-nr_08_O_nr.out.gz` | code/pseudo_dojo/pseudo_dojo/ppcodes/tests/08_O_nr.out | `prog:ONCVPSP-2.1.1:non-relativistic` |
| `oncvpsp-2.1.1-sr_Al-sp.out.gz` | code/pseudo_dojo/pseudo_dojo/pseudos/ONCVPSP-PBE-PDv0.0/Al/Al-sp.out | `prog:ONCVPSP-2.1.1:scalar-relativistic` |
| `oncvpsp-3.0.0-sr_H-ploc-high.out.gz` | code/pseudo_dojo/pseudo_dojo/pseudos/ONCVPSP-PBE-PDv0.0/H/H-ploc-high.out | `nproj_combo:(0, 2)`, `prog:ONCVPSP-3.0.0:scalar-relativistic` |
| `oncvpsp-3.1.1-sr_Ca-sp.out.gz` | code/pseudo_dojo/pseudo_dojo/pseudos/ONCVPSP-PBE-DEV/Ca/Ca-sp.out | `icmod:2`, `prog:ONCVPSP-3.1.1:scalar-relativistic` |
| `oncvpsp-3.2.2-fr_Pt-fsp_r.out.gz` | code/pseudo_dojo/pseudo_dojo/pseudos/ONCVPSP-PBE-DEV/Pt/Pt-fsp_r.out | `prog:ONCVPSP-3.2.2:fully-relativistic` |
| `oncvpsp-3.2.2-sr_C-gw-WORSE.out.gz` | code/pseudo_dojo/pseudo_dojo/pseudos/ONCVPSP-PBE-DEV/C/C-gw-WORSE.out | `nproj_combo:(1,)` |
| `oncvpsp-3.2.2-sr_H.singleProjector.out.gz` | code/pseudo_dojo/pseudo_dojo/pseudos/ONCVPSP-PBE-DEV/H/H.singleProjector.out | `nproj_combo:(0, 1)` |
| `oncvpsp-3.2.3-fr_Og-d_r.out.gz` | code/pseudo_dojo/pseudo_dojo/pseudos/ONCVPSP-PBE-FR-PDv0.3/Og/Og-d_r.out | `fused_atom_line`, `prog:ONCVPSP-3.2.3:fully-relativistic`, `warn:negative_energy_second_projector` |
| `oncvpsp-3.2.3-sr_Sm3+_f-in-core-PBE-icmod1.out.gz` | code/pseudo_dojo/pseudo_dojo/pseudos/ONCVPSP-PBE-PDv0.3/Sm/Sm3+_f-in-core-PBE-icmod1.out | `lpopt:4`, `prog:ONCVPSP-3.2.3:scalar-relativistic` |
| `oncvpsp-3.3.0-fr_La-sp_r.out.gz` | code/pseudo_dojo/pseudo_dojo/pseudos/ONCVPSP-PBE-FR-PDv0.4/La/La-sp_r.out | `prog:ONCVPSP-3.3.0:fully-relativistic`, `warn:wellstate_negative_energy_state` |
| `oncvpsp-3.3.0-sr_Ba-sp-high.out.gz` | code/pseudo_dojo/pseudo_dojo/pseudos/ONCVPSP-PBEsol-PDv0.4/Ba/Ba-sp-high.out | `prog:ONCVPSP-3.3.0:scalar-relativistic` |
| `oncvpsp-3.3.1-sr_F.out.gz` | code/pseudo_dojo/pseudo_dojo/pseudos/ONCVPSP-PW-PDv0.4/F/F_regen_with_PW/F.out | `prog:ONCVPSP-3.3.1:scalar-relativistic` |
| `oncvpsp-4.0.1-fr_74_W_r.out.gz` | oncvpsp/code/oncvpsp/tests/refs/74_W_r.out | `nproj_combo:(2, 3)`, `prog:ONCVPSP-4.0.1:relativistic`, `warn:moderately_localized_bound_state_for_projector` |
| `oncvpsp-4.0.1-fr_N_test_config_failed.out.gz` | oncvpspr.x 4.0.1 on the input of `oncvpsp-4.0.1-sr_N_test_config_failed` (`psfile:upf`) | `test_config_failed:fr`, `binary_in_message` (`psatom_r` writes four integers with the format `(/a,3i4)`; format reversion prints the last one as raw bytes on a line of its own) |
| `oncvpsp-4.0.1-sr_14_Si.out.gz` | data/examples/Si/14_Si.out | `rxpsh:True` |
| `oncvpsp-4.0.1-sr_I_1582.out.gz` | AiiDA calculation a2/f1/1582-ff21-4760-890f-5a69f1443834 | `warn:icmod4_nm_not_converged` |
| `oncvpsp-4.0.1-sr_I_f3c1.out.gz` | AiiDA calculation 30/71/f3c1-3044-4a72-b570-e28366a27058 | `err:ps0norm_gt_uunorm` |
| `oncvpsp-4.0.1-sr_Li-low-00.out.gz` | code/ppego/checkin-2025-03-11/Li-low-00.out | `ncnf:1` |
| `oncvpsp-4.0.1-sr_Li_fd77.out.gz` | AiiDA calculation 71/2a/fd77-f3ac-4f20-b85c-a8d2ca24c6e7 | `err:vkb_bound_state_not_converged` |
| `oncvpsp-4.0.1-sr_N_84c0.out.gz` | AiiDA calculation 20/25/84c0-b486-47ef-932a-243d01159a33 | `err:too_few_qroots`, `warn:scattering_first_projector` |
| `oncvpsp-4.0.1-sr_N_test_config_failed.out.gz` | oncvpsp-opt nitrogen run with 4 test configurations | `ncnf:4`, `test_config_failed:3` (pseudo-atom of test configuration 3 not converged, later configuration completed) |
| `oncvpsp-4.0.1-sr_Pb_143e.out.gz` | AiiDA calculation de/5a/143e-78aa-4671-b9d8-bbc5e4f168a7 | `missing:log_derivatives,gnuplot_script` |
| `oncvpsp-4.0.1-sr_Pb_310a.out.gz` | AiiDA calculation 43/2f/310a-bf93-4d5c-b8c2-f1c4fed92e7b | `err:wellstate_not_converged` |
| `oncvpsp-4.0.1-sr_Pb_4190.out.gz` | AiiDA calculation f7/03/4190-e1bb-44e7-bec1-1a7e65ff7915 | `err:vkbout_dgesv_failed`, `missing:gnuplot_script`, `nld:1` |
| `oncvpsp-4.0.1-sr_Pb_c39f.out.gz` | AiiDA calculation 6b/85/c39f-b90d-42f4-a81a-543092e0abad | `err:wellstate_no_bound_state` |
| `oncvpsp-4.0.1-sr_Pb_d4fb.out.gz` | AiiDA calculation b8/71/d4fb-d6e0-4b3c-a081-9c0424ff0ce2 | `warn:diag_lschvkbb_not_converged`, `warn:first_pswf_has_node` |
| `oncvpsp-4.0.1-sr_pspot_has_node.out.gz` | code/aiida-opsp/test_old/parsers/fixtures/oncv/pspot_has_node/aiida.out | `err:first_pswf_has_node` |
