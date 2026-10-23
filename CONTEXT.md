# pyoncvpsp

Reads and writes ONCVPSP / METAPSP pseudopotential generator inputs, and parses their text outputs into structured data, including where and why a generation failed.

## Language

### Angular momentum

**l-channel**:
The set of pseudopotential quantities for one orbital angular momentum `l`, with no total angular momentum `j` defined. Input parameters are always given per l-channel; non- and scalar-relativistic calculations have only l-channels, and fully-relativistic outputs also report scalar and spin-orbit recombinations of j-channels per l-channel.
_Avoid_: channel (unqualified), ell

**j-channel**:
The set of pseudopotential quantities for one pair `(l, j)` with `j = l ± 1/2`, defined only in fully-relativistic calculations, which split each l-channel from the input into its j-channels. The `l = 0` l-channel splits into the single j-channel `(0, 1/2)`.
_Avoid_: spin channel, spin, channel (unqualified)

### Projectors

**Projector**:
The `iproj`-th Kleinman-Bylander projector of an l-channel or j-channel, identified by `(channel, iproj)`. The same projector is first optimized in the semilocal stage, then constructed in the VKB stage.
_Avoid_: VKB projector (when the projector itself is meant), optimized projector

**Optimized pseudo-wavefunction**:
The pseudo-wavefunction optimized for one projector in the semilocal stage, together with its convergence profile; the VKB stage builds the projector from it.
_Avoid_: optimized projector (except when quoting ONCVPSP output)

**Valence state**:
A state of the reference configuration listed after the core states. Its occupation is always positive (the programs reject `f <= 0`), but may be tiny.
_Avoid_: occupied state, bound state (as a synonym)

**Projector state**:
The all-electron reference state a projector is built from: a valence state, a wellstate, or a localized bound state.

**Wellstate**:
An unbound projector state, solved in a well potential added beyond rc so that it is normalizable; in azadoks/oncvpsp its tail may instead be the free solution or that of a confining wall.
_Avoid_: unbound state

**Localized bound state**:
An unoccupied bound projector state that is localized enough to be used in place of a wellstate.
_Avoid_: negative-energy state, normal bound state

### Configurations

**Reference configuration**:
The atomic configuration the pseudopotential is generated for; its states after the core are the valence states.

**Test configuration**:
An atomic configuration other than the reference configuration, against which the finished pseudopotential is tested.
_Avoid_: configuration 0 (for the reference configuration)

### Output structure

**Stage**:
One step of the program, in execution order (input, all-electron reference, semilocal, VKB, VKB solve, model core, diagnostics, test configurations, log derivatives and plots, output). Output is printed in stage order.
_Avoid_: section, phase

**Block**:
A stretch of output inside one stage, opened by a header line and covering one projector, l-channel, j-channel or configuration; it ends where the next block or stage begins.
_Avoid_: section

### Runs and messages

**Message**:
A recognized diagnostic that ONCVPSP or METAPSP prints to stdout, matched by a message definition.
_Avoid_: log line, error summary

**Error**:
A message after which the program stops, whatever its printed wording.
_Avoid_: fatal error, fatal message

**Warning**:
A message after which the program continues, even if its printed text says "ERROR".

**Completed run**:
A run that wrote all of its plotting data.
_Avoid_: successful run

**Failure**:
A run that did not complete. Its cause is the first error, or incomplete output if no error was printed.
_Avoid_: crash, failed run (for a completed run with errors in its text)
