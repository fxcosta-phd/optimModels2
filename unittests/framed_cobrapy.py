from pathlib import Path
import time
import warnings

from cobra.io import read_sbml_model
import pytest
from reframed.io.sbml import load_cbmodel

from optimModels.optimization.evaluation_functions import build_evaluation_function
from optimModels.optimization.run import cbm_strain_optim
from optimModels.simulation.simul_problems import StoicSimulationProblem

REPO_ROOT = Path(__file__).resolve().parents[1]
SBML_FILE = REPO_ROOT / "examples" / "models" / "Ec_iAF1260.xml"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

BIOMASS_RXN = "Ec_biomass_iAF1260_core_59p81M"
PRODUCT_RXN = "EX_succ_e_"
SUBSTRATE_RXN = "EX_glc_e_"


def print_results(file_path, population, optim_config):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "population_size;candidate_max_size;crossover_rate;mutation_rate;new_candidates_rate;num_elites\n"
        )
        handle.write(";".join(map(str, optim_config.get_ea_configurations().get_default_config())) + "\n")
        handle.write("Generation;Fitness;Candidate;Reactions\n")

        for ind in population:
            solution_decoded = optim_config.getdecoder().decode_candidate(ind.candidate)
            handle.write(f"100;{ind.fitness};{ind.candidate};{solution_decoded}\n")


def framed_optim(is_multiproc, size, max_generations=None):
    model = load_cbmodel(str(SBML_FILE), flavor="cobra")
    biomass_id = BIOMASS_RXN if BIOMASS_RXN in model.reactions else f"R_{BIOMASS_RXN}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_res = RESULTS_DIR / "optim_Ec_iAF1260_ko.csv"

    simul_prob = StoicSimulationProblem(model, objective={biomass_id: 1})
    obj_func = build_evaluation_function("targetFlux", [biomass_id])

    kwargs = {"maxGenerations": max_generations} if max_generations is not None else {}
    cbm_strain_optim(
        simul_prob,
        evaluationFunc=obj_func,
        levels=None,
        isMultiProc=is_multiproc,
        candidateSize=size,
        resultFile=str(file_res),
        **kwargs,
    )


def cobra_optim(is_multiproc, size, max_generations=None):
    model = read_sbml_model(str(SBML_FILE))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_res = RESULTS_DIR / "optim_Ec_iAF1260_ko_cobra.csv"

    simul_prob = StoicSimulationProblem(model, objective={BIOMASS_RXN: 1}, withCobraPy=True)
    obj_func = build_evaluation_function("targetFlux", [BIOMASS_RXN])

    print(simul_prob.simulate())
    kwargs = {"maxGenerations": max_generations} if max_generations is not None else {}
    cbm_strain_optim(
        simul_prob,
        evaluationFunc=obj_func,
        levels=None,
        isMultiProc=is_multiproc,
        candidateSize=size,
        resultFile=str(file_res),
        **kwargs,
    )


def mewpy_optim(is_multiproc, size, max_generations=1):
    """Modern replacement for the legacy cameo ReactionKnockoutOptimization benchmark."""
    from mewpy.optimization import EA
    from mewpy.optimization.evaluation import BPCY
    from mewpy.problems import RKOProblem

    model = load_cbmodel(str(SBML_FILE), flavor="cobra")
    biomass_id = BIOMASS_RXN if BIOMASS_RXN in model.reactions else f"R_{BIOMASS_RXN}"
    product_id = PRODUCT_RXN if PRODUCT_RXN in model.reactions else f"R_{PRODUCT_RXN}"

    evaluator = BPCY(biomass_id, product_id)
    problem = RKOProblem(model, fevaluation=[evaluator], candidate_max_size=size)
    ea = EA(problem, max_generations=max_generations, mp=is_multiproc)
    final_pop = ea.run()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ea.dataframe().to_csv(RESULTS_DIR / "optim_Ec_iAF1260_ko_mewpy.csv", index=False)
    return final_pop


def print_drains_cobrapy():
    model = read_sbml_model(str(SBML_FILE))
    for rxn in model.exchanges:
        print(f"{rxn.id}: {rxn.lower_bound} <---> {rxn.upper_bound}")


@pytest.mark.skipif(not SBML_FILE.exists(), reason=f"Benchmark model not found at {SBML_FILE}")
def test_framed_and_cobrapy_imports():
    """Verify models load cleanly in both reframed and cobrapy."""
    reframed_model = load_cbmodel(str(SBML_FILE), flavor="cobra")
    cobra_model = read_sbml_model(str(SBML_FILE))
    assert len(reframed_model.reactions) > 0
    assert len(cobra_model.reactions) > 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    candidate_size = 6

    t1 = time.time()
    framed_optim(False, candidate_size)
    print(f"time of REFRAMED: {time.time() - t1:.2f}s")

    t1 = time.time()
    cobra_optim(False, candidate_size)
    print(f"time of COBRAPY: {time.time() - t1:.2f}s")