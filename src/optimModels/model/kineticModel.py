from collections import OrderedDict
from graphlib import TopologicalSorter
import math
import re
import libsbml
from reframed.core.model import Compartment, Metabolite, Model, Reaction


def load_kinetic_model(filename, param_map=None):
    """
    Load a dynamic kinetic model from an SBML file using libSBML and reframed's Model base.

    Args:
        filename (str): Path to the SBML file.
        param_map (dict | str | None): Mapping of {reaction_id: [param1, param2]} or a regex
            pattern used to extract modifiable parameters from rate laws.

    Returns:
        KineticModel
    """
    document = libsbml.readSBMLFromFile(filename)
    sbml_model = document.getModel()

    if sbml_model is None:
        raise IOError(f"Failed to load SBML model from '{filename}'.")

    model = KineticModel(sbml_model.getId() or "kinetic_model")

    # 1. Compartments
    for comp in sbml_model.getListOfCompartments():
        c_id = comp.getId()
        model.add_compartment(Compartment(c_id, name=comp.getName() or c_id))
        if comp.isSetSize():
            model.compartments[c_id].size = comp.getSize()
        else:
            model.compartments[c_id].size = 1.0

    # 2. Metabolites (Species) & Initial Concentrations
    for species in sbml_model.getListOfSpecies():
        s_id = species.getId()
        comp_id = species.getCompartment()
        model.add_metabolite(Metabolite(s_id, name=species.getName() or s_id, compartment=comp_id))

        if species.isSetInitialConcentration():
            conc = species.getInitialConcentration()
        elif species.isSetInitialAmount():
            comp_size = model.compartments[comp_id].size if comp_id in model.compartments else 1.0
            conc = species.getInitialAmount() / comp_size
        else:
            conc = 0.0

        model.concentrations[s_id] = conc
        if species.getBoundaryCondition() or species.getConstant():
            model.boundary_metabolites.add(s_id)

    # 3. Global Parameters
    for param in sbml_model.getListOfParameters():
        p_id = param.getId()
        val = param.getValue() if param.isSetValue() else 0.0
        if param.getConstant():
            model.global_params[p_id] = val
        else:
            model.variable_params[p_id] = val

    # 4. Reactions, Local Parameters & Rate Laws
    for rxn in sbml_model.getListOfReactions():
        r_id = rxn.getId()
        stoichiometry = OrderedDict()

        for reactant in rxn.getListOfReactants():
            m_id = reactant.getSpecies()
            stoich = reactant.getStoichiometry() if reactant.isSetStoichiometry() else 1.0
            stoichiometry[m_id] = stoichiometry.get(m_id, 0.0) - stoich

        for product in rxn.getListOfProducts():
            m_id = product.getSpecies()
            stoich = product.getStoichiometry() if product.isSetStoichiometry() else 1.0
            stoichiometry[m_id] = stoichiometry.get(m_id, 0.0) + stoich

        model.add_reaction(
            Reaction(
                r_id,
                name=rxn.getName() or r_id,
                reversible=rxn.getReversible(),
                stoichiometry=stoichiometry,
            )
        )

        if rxn.isSetKineticLaw():
            kl = rxn.getKineticLaw()
            for l_param in kl.getListOfParameters():
                lp_id = l_param.getId()
                val = l_param.getValue() if l_param.isSetValue() else 0.0
                model.local_params[f"{r_id}_{lp_id}"] = val

            formula = libsbml.formulaToL3String(kl.getMath()) if kl.isSetMath() else "0.0"
            model.ratelaws[r_id] = formula
        else:
            model.ratelaws[r_id] = "0.0"

    # 5. Assignment Rules
    for rule in sbml_model.getListOfRules():
        if rule.isAssignment():
            var_id = rule.getVariable()
            formula = libsbml.formulaToL3String(rule.getMath()) if rule.isSetMath() else "0.0"
            model.assignment_rules[var_id] = formula
            model.variable_params.setdefault(var_id, 0.0)

    # Parse expressions into Python syntax
    model._set_parsed_attr()

    if isinstance(param_map, str):
        extracted = OrderedDict(
            (r_id, re.findall(param_map, ratelaw))
            for r_id, ratelaw in model.ratelaws.items()
        )
        model.reac_params_factors = OrderedDict(
            (r_id, params) for r_id, params in extracted.items() if params
        )
    else:
        model.set_reactions_parameters_factors(param_map)

    return model


def _piecewise(*args):
    """Helper to evaluate SBML piecewise(val1, cond1, ..., otherwise) expressions."""
    for i in range(0, len(args) - 1, 2):
        if args[i + 1]:
            return args[i]
    return args[-1] if len(args) % 2 == 1 else 0.0


class KineticModel(Model):
    """
    Extension of reframed's Model class for dynamic ODE simulation and strain optimization.
    Supports knockouts, under-expression, and over-expression via parameter or enzyme scaling factors.
    """

    def __init__(self, model_id):
        super().__init__(model_id)
        self.concentrations = OrderedDict()
        self.boundary_metabolites = set()
        self.global_params = OrderedDict()
        self.local_params = OrderedDict()
        self.variable_params = OrderedDict()
        self.ratelaws = OrderedDict()
        self.assignment_rules = OrderedDict()

        self.reac_params_factors = OrderedDict()
        self.parsed_rates = OrderedDict()
        self.parsed_rules = OrderedDict()
        self.parsed_xdot = OrderedDict()
        self._met_index = {}

    def merge_constants(self):
        """Combine compartment sizes, global constants, and local reaction parameters."""
        constants = OrderedDict()
        for c_id, comp in self.compartments.items():
            constants[c_id] = getattr(comp, "size", 1.0)
        constants.update(self.global_params)
        constants.update(self.local_params)
        return constants

    def get_initial_concentrations(self):
        """Return initial metabolite concentrations ordered to match the ODE state vector x."""
        return [self.concentrations.get(m_id, 0.0) for m_id in self.metabolites]

    def _parse_expression(self, expr, rxn_id=None):
        """Convert an SBML math string into a valid Python expression referencing x, v, and p."""
        py_expr = expr.replace("^", "**")

        def _replace_token(match):
            token = match.group(0)
            # Leave math function calls (e.g., exp(...), pow(...)) untouched
            if match.end() < len(py_expr) and py_expr[match.end()] == "(":
                return token
            if token == "time":
                return "t"
            if rxn_id and f"{rxn_id}_{token}" in self.local_params:
                return f"p['{rxn_id}_{token}']"
            if token in self._met_index:
                return f"x[{self._met_index[token]}]"
            if token in self.assignment_rules or token in self.variable_params:
                return f"v['{token}']"
            if token in self.global_params or token in self.compartments:
                return f"p['{token}']"
            return token

        return re.sub(r"\b[A-Za-z_]\w*\b", _replace_token, py_expr)

    def _set_parsed_attr(self):
        self._met_index = {m_id: idx for idx, m_id in enumerate(self.metabolites.keys())}

        # 1. Parse and topologically sort assignment rules
        raw_parsed_rules = {
            var_id: self._parse_expression(rule)
            for var_id, rule in self.assignment_rules.items()
        }
        dep_graph = {
            var_id: set(re.findall(r"v\['(.*?)'\]", parsed))
            for var_id, parsed in raw_parsed_rules.items()
        }
        ordered_rule_ids = list(TopologicalSorter(dep_graph).static_order())
        self.parsed_rules = OrderedDict(
            (var_id, raw_parsed_rules[var_id])
            for var_id in ordered_rule_ids
            if var_id in raw_parsed_rules
        )

        # 2. Parse reaction rate laws
        self.parsed_rates = OrderedDict(
            (r_id, self._parse_expression(ratelaw, rxn_id=r_id))
            for r_id, ratelaw in self.ratelaws.items()
        )

        # 3. Build mass balance expressions (dx/dt) for non-boundary metabolites
        self.parsed_xdot = OrderedDict()
        for m_id in self.metabolites.keys():
            if m_id in self.boundary_metabolites:
                self.parsed_xdot[m_id] = "0.0"
                continue

            terms = []
            for r_id, rxn in self.reactions.items():
                coeff = rxn.stoichiometry.get(m_id, 0.0)
                if coeff == 0.0:
                    continue
                if coeff == 1.0:
                    terms.append(f"+ r['{r_id}']")
                elif coeff == -1.0:
                    terms.append(f"- r['{r_id}']")
                elif coeff > 0:
                    terms.append(f"+ {coeff} * r['{r_id}']")
                else:
                    terms.append(f"- {abs(coeff)} * r['{r_id}']")

            expr = " ".join(terms).lstrip("+ ") if terms else "0.0"
            self.parsed_xdot[m_id] = expr

    def build_ode(self, factors=None):
        """
        Generate the Python source code for the ODE system with optional scaling factors.

        Args:
            factors (dict | None): {param_or_metabolite_id: scale_factor}, where 0 = knockout,
                (0, 1) = under-expression, and > 1 = over-expression.

        Returns:
            str: Python function source code for `ode_func(t, x, r, p, v)`.
        """
        factors = factors or {}
        factors_enz = {k: val for k, val in factors.items() if k in self.metabolites}
        factors_param = {k: val for k, val in factors.items() if k not in factors_enz}

        rule_exprs = [
            f"    v['{p_id}'] = {expr}"
            for p_id, expr in self.parsed_rules.items()
        ]

        rate_exprs = []
        for r_id in self.reactions.keys():
            new_exp = self.parsed_rates[r_id]
            if r_id in self.reac_params_factors:
                to_modify = set(factors_param.keys()).intersection(self.reac_params_factors[r_id])
                for elem in to_modify:
                    scale = factors_param[elem]
                    # Match both global param IDs and local '{r_id}_{elem}' param IDs
                    pattern = rf"([pv]\['(?:{re.escape(r_id)}_)?{re.escape(elem)}'\])"
                    new_exp = re.sub(pattern, rf"{scale} * \1", new_exp)
            rate_exprs.append(f"    r['{r_id}'] = {new_exp}")

        balances = []
        for m_id in self.metabolites.keys():
            exp = self.parsed_xdot[m_id]
            if m_id in factors_enz:
                scale = factors_enz[m_id]
                new_exp = "0.0" if scale == 0 else f"({exp}) * {scale}"
                balances.append(f"        {new_exp}")
            else:
                balances.append(f"        {exp}")

        return (
            "def ode_func(t, x, r, p, v):\n\n"
            + "\n".join(rule_exprs)
            + "\n\n"
            + "\n".join(rate_exprs)
            + "\n\n"
            + "    dxdt = [\n"
            + ",\n".join(balances)
            + "\n    ]\n\n"
            + "    return dxdt\n"
        )

    def get_ode(self, r_dict=None, params=None, factors=None, abstol=1e-9):
        """
        Compile and return a callable `f(t, x)` for numerical ODE integration.

        Args:
            r_dict (dict | None): Optional dictionary populated with reaction rates at each step.
            params (dict | None): Parameter overrides {param_id: new_value}.
            factors (dict | None): Scaling factors for parameters or enzyme species.
            abstol (float): Tolerance below which small negative concentrations are clamped to 0.

        Returns:
            Callable[[float, Sequence[float]], list[float]]
        """
        p = self.merge_constants()
        v = self.variable_params.copy()
        r = r_dict if r_dict is not None else {}

        if params:
            p.update(params)

        exec_globals = {
            "__builtins__": {},
            "math": math,
            "exp": math.exp,
            "log": math.log,
            "ln": math.log,
            "log10": math.log10,
            "sqrt": math.sqrt,
            "pow": math.pow,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "abs": abs,
            "min": min,
            "max": max,
            "piecewise": _piecewise,
        }
        local_ns = {}
        exec(self.build_ode(factors), exec_globals, local_ns)
        compiled_ode = local_ns["ode_func"]

        def f(t, x):
            x_clean = list(x)
            for idx, val in enumerate(x_clean):
                if val < 0.0:
                    if val >= -abstol:
                        x_clean[idx] = 0.0
                    else:
                        raise ValueError(
                            f"Significant negative concentration ({val}) encountered at index {idx}."
                        )
            return compiled_ode(t, x_clean, r, p, v)

        return f

    def set_reactions_parameters_factors(self, param_map):
        """Set the modifiable parameters per reaction: {reaction_id: [param1, param2]}."""
        self.reac_params_factors = OrderedDict(param_map) if param_map else OrderedDict()