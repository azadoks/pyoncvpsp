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
| `oncvpsp-4.0.1-nr_08_O_build_info.out.gz` | `oncvpspnr.x` of azadoks/oncvpsp 1f11258 (branch `oncvpsp-opt`, then named `ae-pseudo-split`) on `tests/data/08_O.dat` | `prog:ONCVPSP-4.0.1:non-relativistic`, `build_line` (the header says scalar-relativistic; the build line gives the executable and its relativistic treatment) |
| `oncvpsp-4.0.1-sr_14_Si_bc.out.gz` | `oncvpsp.x` of azadoks/oncvpsp a4046c0 (branch `oncvpsp-opt`) on its `tests/data/14_Si_bc.dat` | `icmod:6` (budgeted Martins model core) |
| `oncvpsp-4.0.1-sr_14_Si_bca.out.gz` | `oncvpsp.x` of azadoks/oncvpsp f098dc9 (branch `icmod6-auto`) on its `tests/data/14_Si_bca.dat` | `budgeted_core:ecut_pseudo_waves` (icmod 6 budget taken from the pseudo-waves, `fcfact` 0) |
| `oncvpsp-4.0.1-sr_14_Si_log_mesh.out.gz` | `oncvpsp.x` of azadoks/oncvpsp 8490a59 (branch `radial-mesh-input`) on its `tests/data/14_Si.dat` with the log mesh line `1.004 -1 50.0` | `log_mesh` (the mesh set by the input, echoed after the test configurations and in the psp8 input) |
| `oncvpsp-4.0.1-sr_14_Si_new_options.out.gz` | `oncvpsp.x` of azadoks/oncvpsp 522b58a (branch `oncvpsp-opt`; frozen at ~/.local/share/oncvpsp-nist-522b58a) on its sample `14_Si_new_options.dat`: `psfile` upf, `iexc` 7, `c_light` 137.0359895, `upf_mesh` 3500 and the log mesh line | `iexc:7` (VWN with relativistic exchange, UPF `RXC VWN`), `c_light` (seventh value of the atom line, echoed when not the default), `upf_mesh` (third value of the output grid line; the UPF file on the log mesh), `ae_reference_energies:terms` (screening-potential residual and energy terms of the all-electron atom) |
| `oncvpsp-4.0.1-sr_14_Si_ae.out.gz` | `oncvpsp_ae.x -a si.ae` of azadoks/oncvpsp 5947faf (branch `oncvpsp-opt`) on its `tests/data/14_Si.dat` | `stage:ae` (all-electron stage: input and all-electron sections, then the data file it wrote) |
| `oncvpsp-4.0.1-sr_14_Si_ps.out.gz` | `oncvpsp_ps.x -a si.ae` of the same build on the same input, reading that file | `stage:ps` (prints what `oncvpsp.x` prints, the build line aside) |
| `oncvpsp-4.0.1-sr_14_Si_icmod6_auto.out.gz` | `oncvpsp.x` of azadoks/oncvpsp 9b90162 (branch `oncvpsp-opt`, model core v2) on its `tests/data/14_Si_bca.dat` | `budgeted_core:minimum_tail` (icmod 6 with the minimum-tail fill and its coefficients) |
| `oncvpsp-4.0.1-sr_14_Si_icmod7_slack0.7.out.gz` | `oncvpsp.x` of the same build on its `tests/data/14_Si_hm.dat` with slack 0.7 | `icmod:7` (hardness-matched core, its candidates), `slack` (input echo and output) |
| `oncvpsp-4.0.1-fr_14_Si_icmod7.out.gz` | `oncvpspr.x` of the same build on its `tests/data/14_Si_hm.dat` | `icmod:7` fully relativistic (J-averaged probes) |
| `oncvpsp-4.0.1-sr_14_Si_icmod7_tables.out.gz` | `oncvpsp.x` of azadoks/oncvpsp 59d0ba8 (branch `oncvpsp-opt`, table budget) on its `tests/data/14_Si_hm.dat` (icmod 7, fcfact 0, 1e-5, slack 0.7) | `budgeted_core:budget_source:tables` (the budget is the largest Ecut of the convergence tables) |
| `oncvpsp-4.0.1-sr_07_N_opt_diagnostics.out.gz` | `oncvpsp.x` of azadoks/oncvpsp 99e97fc (branch `smooooth`, X-3) on its `tests/data/07_N_opt.dat` (the oncvpsp-opt N baseline: PBE via Libxc, icmod 6, lpopt 5, UPF on the log mesh) | `diagnostics` (the DIAGNOSTICS block: Ecut crossings, log derivatives and ghost-test margins to 17 digits) |
| `oncvpsp-4.0.1-fr_14_Si_diagnostics.out.gz` | `oncvpspr.x` of the same build on its `tests/data/14_Si.dat` | `diagnostics` fully relativistic (ikap 1 and 2 as j = l ± 1/2) |
| `oncvpsp-4.0.1-sr_08_O_ionic_tail.out.gz` | `oncvpsp.x` of azadoks/oncvpsp 5e706ad (branch `oncvpsp-opt`, ionic tail) on the PseudoDojo v0.6 O input (icmod 3, rlmax 6, no test configurations) | `ionic_tail`, `coulomb_tail` (V_loc switched to the frozen-core ionic tail, and its Coulomb-tail diagnostics) |
| `oncvpsp-4.0.1-fr_74_W_ionic_tail.out.gz` | `oncvpspr.x` of the same build on a W input from the oncvpsp test set (sample of the ionic-tail port) | `ionic_tail` fully relativistic |
| `oncvpsp-4.0.1-sr_N_budget_cap.out.gz` | `oncvpsp.x` of azadoks/oncvpsp a4046c0 on N with rc 3.2, icmod 6, cutoff budget 20 Ha | `budgeted_core:r_core_max` (rc beyond the last radius with core charge) |
| `oncvpsp-4.0.1-sr_N_budget_too_small.out.gz` | `oncvpsp.x` of azadoks/oncvpsp a4046c0 on N with rc 1.5, icmod 6, cutoff budget 2 Ha | `err:modcore_budget_no_model` |
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
