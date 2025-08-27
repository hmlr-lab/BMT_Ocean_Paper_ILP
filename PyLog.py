import re
from collections import defaultdict

from dataclasses import dataclass
import PyGol as pygol


@dataclass
class ProofResult:
    stat: bool
    sub: str
    proof: any


# --------------------- Term Structures ----------------------
class Variable:
    __slots__ = ["name"]
    def __init__(self, name): self.name = name
    def __repr__(self): return f"Var({self.name})"
    def __eq__(self, other): return isinstance(other, Variable) and self.name == other.name
    def __hash__(self): return hash(self.name)

class Functor:
    __slots__ = ["name", "args"]
    def __init__(self, name, args=None):
        self.name = name
        self.args = args or []
    def __repr__(self):
        return f"{self.name}({', '.join(repr(a) for a in self.args)})" if self.args else self.name
    def arity(self): return len(self.args)
    def __eq__(self, other):
        return isinstance(other, Functor) and self.name == other.name and self.args == other.args
    def __hash__(self): return hash((self.name, tuple(self.args)))

# --------------------- Parsing ----------------------
def parse_term(term_str):
    term_str = term_str.strip()
    # Variable?
    if re.match(r'^[_A-Z]\w*$', term_str):
        return Variable(term_str)
    # Try as number
    try:
        if '.' in term_str:
            return Functor(str(float(term_str)), [])
        else:
            return Functor(str(int(term_str)), [])
    except ValueError:
        pass
    # Python special call
    if term_str.startswith('!py!'):
        name_and_args = term_str[4:]
        func_match = re.match(r'^(\w+)\((.*)\)$', name_and_args)
        if not func_match:
            raise ValueError(f"Malformed Python call: {term_str}")
        fname, argstr = func_match.groups()
        arg_list = split_args(argstr)
        parsed_args = [parse_term(a.strip()) for a in arg_list]
        return Functor('!py!', [Functor(fname, parsed_args)])
    # Functor or atom
    match = re.match(r'^([a-z0-9_<>=+\-*/]+)(?:\((.*)\))?$', term_str, re.DOTALL)
    if not match: raise ValueError(f"Cannot parse term: {term_str}")
    name, args_str = match.group(1), match.group(2)
    if args_str is None:
        return Functor(name, [])
    arg_list = split_args(args_str)
    parsed_args = [parse_term(a.strip()) for a in arg_list]
    return Functor(name, parsed_args)

def split_args(args_str):
    args, bracket_depth, current = [], 0, []
    for ch in args_str:
        if ch == '(':
            bracket_depth += 1
            current.append(ch)
        elif ch == ')':
            bracket_depth -= 1
            current.append(ch)
        elif ch == ',' and bracket_depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current: args.append("".join(current).strip())
    return [a.strip() for a in args]

def parse_clause(clause_str):
    clause_str = clause_str.strip()
    if clause_str.endswith('.'): clause_str = clause_str[:-1].strip()
    if ':-' in clause_str:
        head_part, body_part = clause_str.split(':-', 1)
        head = parse_term(head_part.strip())
        body_terms = [parse_term(p.strip()) for p in split_args(body_part.strip())]
        return (head, body_terms)
    else:
        head = parse_term(clause_str)
        return (head, [])

# --------------------- Unification ----------------------
def walk(term, subs):
    while isinstance(term, Variable) and term in subs: term = subs[term]
    return term

def unify(term1, term2, subs=None):
    if subs is None: subs = {}
    term1, term2 = walk(term1, subs), walk(term2, subs)
    if term1 == term2: return subs
    if isinstance(term1, Variable): return bind_variable(term1, term2, subs)
    if isinstance(term2, Variable): return bind_variable(term2, term1, subs)
    if isinstance(term1, Functor) and isinstance(term2, Functor):
        if (term1.name != term2.name) or (len(term1.args) != len(term2.args)): return None
        for a1, a2 in zip(term1.args, term2.args):
            subs = unify(a1, a2, subs)
            if subs is None: return None
        return subs
    return None

def bind_variable(var, val, subs):
    if occur_check(var, val, subs): return None
    new_subs = dict(subs); new_subs[var] = val; return new_subs

def occur_check(var, term, subs):
    term = walk(term, subs)
    if var == term: return True
    if isinstance(term, Functor): return any(occur_check(var, a, subs) for a in term.args)
    return False

# --------------------- Python Function Call Helper ----------------------
def get_py_value(term, subs, kb):
    term = walk(term, subs)
    if isinstance(term, Functor):
        if term.name == '!py!':
            inner = term.args[0]
            if inner.name in kb.py_functions:
                py_args = [get_py_value(a, subs, kb) for a in inner.args]
                return kb.py_functions[inner.name](*py_args)
            else:
                raise ValueError(f"Unknown py function: {inner.name}")
        if not term.args:
            try:
                return float(term.name)
            except ValueError:
                return term.name
        else:
            return tuple(get_py_value(a, subs, kb) for a in term.args)
    elif isinstance(term, Variable):
        raise ValueError(f"Unbound variable in python call: {term}")
    else:
        return term
    



import uuid

def standardize_apart_pair(head, body):
    """
    Given a rule (head, body), return (fresh_head, fresh_body)
    where all variables are replaced with unique ones,
    and the same variable in head/body is the same new variable.
    """
    mapping = {}
    def _std(term):
        if isinstance(term, Variable):
            if term not in mapping:
                mapping[term] = Variable(f"{term.name}_{uuid.uuid4().hex}")
            return mapping[term]
        elif isinstance(term, Functor):
            return Functor(term.name, [_std(a) for a in term.args])
        else:
            return term
    return _std(head), [_std(b) for b in body]

# --------------------- Built-in Arithmetic/Comparison + !py! ----------------------
def evaluate_arithmetic(functor, subs, kb=None):
    name = functor.name
    args = [walk(a, subs) for a in functor.args]
    def to_number(t):
        if isinstance(t, Functor) and len(t.args) == 0:
            try: return float(t.name)
            except ValueError: return t
        return t
    args = [to_number(a) for a in args]
    if name == "is" and len(args) == 2:
        rhs_val = None
        # If rhs is !py!
        if isinstance(functor.args[1], Functor) and functor.args[1].name == '!py!':
            inner = functor.args[1].args[0]
            if kb is not None and inner.name in kb.py_functions:
                py_args = [get_py_value(a, subs, kb) for a in inner.args]
                rhs_val = kb.py_functions[inner.name](*py_args)
        else:
            rhs_val = evaluate_expression(functor.args[1], subs, kb)
        if rhs_val is None: return None
        val_functor = Functor(str(rhs_val), [])
        return unify(functor.args[0], val_functor, subs)
    if name in ("<", "=<", ">", ">=", "=:=", "=\=", "==") and len(args) == 2:
        left, right = args
        if not (isinstance(left, float) and isinstance(right, float)): return None
        if name == "<": return subs if (left < right) else None
        elif name == "=<": return subs if (left <= right) else None
        elif name == ">": return subs if (left > right) else None
        elif name == ">=": return subs if (left >= right) else None
        elif name == "=:=": return subs if (abs(left - right) < 1e-9) else None
        elif name == "=\=": return subs if (abs(left - right) >= 1e-9) else None
        elif name == "==": return subs if abs(left - right) < 1e-9 else None

    return None

def evaluate_expression(term, subs, kb=None):
    term = walk(term, subs)
    if isinstance(term, Functor) and not term.args:
        try: return float(term.name)
        except ValueError: return None
    if isinstance(term, Functor) and term.name == '!py!':
        inner = term.args[0]
        if kb is not None and inner.name in kb.py_functions:
            py_args = [get_py_value(a, subs, kb) for a in inner.args]
            return kb.py_functions[inner.name](*py_args)
    if isinstance(term, Functor) and term.arity() == 2:
        op = term.name
        left_val = evaluate_expression(term.args[0], subs, kb)
        right_val = evaluate_expression(term.args[1], subs, kb)
        if left_val is None or right_val is None: return None
        if op == "+": return left_val + right_val
        elif op == "-": return left_val - right_val
        elif op == "*": return left_val * right_val
        elif op == "/":
            if abs(right_val) < 1e-9: return None
            return left_val / right_val
    return None

def try_arithmetic_builtin(goal, subs, kb=None):
    if not isinstance(goal, Functor): return None
    return evaluate_arithmetic(goal, subs, kb=kb)

def apply_substitution(term, subs):
    term = walk(term, subs)
    if isinstance(term, Functor):
        new_args = [apply_substitution(a, subs) for a in term.args]
        return Functor(term.name, new_args)
    return term

# --------------------- Helper to pretty-print Functor/Variable ---------------------
def functor_to_str(f):
    if isinstance(f, Variable):
        return f.name
    elif isinstance(f, Functor):
        if f.args:
            return f"{f.name}({', '.join(functor_to_str(a) for a in f.args)})"
        else:
            return f"{f.name}"
    else:
        return str(f)

# --------------------- Knowledge Base Engine with Proof Paths and Clause Failure ----------------------
class KnowledgeBase:
    def __init__(self):
        self.rules = defaultdict(list)
        self.py_functions = {}
    def register_py_function(self, name, fn):
        self.py_functions[name] = fn
    def add_clause(self, clause_str):
        head, body = parse_clause(clause_str)
        key = (head.name, head.arity())
        self.rules[key].append((head, body))
    def add_clauses(self, clauses):
        if isinstance(clauses, str): clauses = [clauses]
        for c in clauses: self.add_clause(c)

    def query(self, query_str, all_solutions=True, show_path=True, explain_failure=True):
        query_term = parse_term(query_str)
        solutions = []
        found_any = False
        for sol, proof in self._solve([query_term], {}, [], show_path=show_path, explain_failure=explain_failure):
            found_any = True
            if show_path:
                solutions.append((sol, proof, "success"))
            else:
                solutions.append(sol)
            if not all_solutions: break
        if not found_any and explain_failure:
            for sol, proof in self._solve([query_term], {}, [], show_path=True, explain_failure=True, explain_on_fail_only=True):
                solutions.append((sol, proof, "failure"))
        return solutions

    """def _solve(self, goals, subs, proof, show_path=False, explain_failure=False, explain_on_fail_only=False):
        if not goals:
            yield subs, proof
            return
        first, *rest = goals

        # Negation as failure: not(Goal)
        if isinstance(first, Functor) and first.name == 'not' and len(first.args) == 1:
            neg_goal = first.args[0]
            found = False
            for _, _ in self._solve([neg_goal], dict(subs), proof, show_path, explain_failure):
                found = True
                break
            if not found:
                new_proof = proof + [("not", functor_to_str(neg_goal))]
                yield from self._solve(rest, subs, new_proof, show_path, explain_failure)
            else:
                if explain_failure and explain_on_fail_only:
                    yield None, proof + [("negation_failed", functor_to_str(neg_goal))]
            return

        # Arithmetic built-ins and Python function assignment
        built_in_subst = try_arithmetic_builtin(first, subs, self)
        if built_in_subst is not None:
            new_proof = proof + [("builtin", functor_to_str(first))]
            yield from self._solve(rest, built_in_subst, new_proof, show_path, explain_failure)
            return

        if not isinstance(first, Functor):
            if explain_failure and explain_on_fail_only:
                yield None, proof + [("fail_not_functor", functor_to_str(first))]
            return
        key = (first.name, first.arity())
        if key not in self.rules:
            if explain_failure and explain_on_fail_only:
                yield None, proof + [("no_matching_fact_or_rule", functor_to_str(first))]
            return

        matched_any = False
        for (rule_head, rule_body) in self.rules[key]:
            new_subs = unify(first, rule_head, dict(subs))
            if new_subs is not None:
                matched_any = True
                step = ("rule", functor_to_str(rule_head), [functor_to_str(b) for b in rule_body])
                # Try proving body goals one by one, and explain where it fails
                body_goals = [apply_substitution(b, new_subs) for b in rule_body]
                rest_goals = [apply_substitution(r, new_subs) for r in rest]

                current_subs = new_subs
                current_proof = proof + [step]
                success = True
                for i, g in enumerate(body_goals):
                    found_body = False
                    for s, p in self._solve([g], current_subs, [], show_path, explain_failure, explain_on_fail_only):
                        if s is not None:
                            current_subs = s
                            current_proof += p
                            found_body = True
                            break
                    if not found_body:
                        fail_msg = (
                            "failed_body_goal",
                            f"In clause {functor_to_str(rule_head)} :- ... , the subgoal #{i+1} `{functor_to_str(g)}` failed.",
                            f"bindings: {pretty_substitution(current_subs)}"
                        )
                        full_proof = current_proof + [fail_msg]
                        if explain_failure and explain_on_fail_only:
                            yield None, full_proof
                        success = False
                        break
                if success:
                    for s, p in self._solve(rest_goals, current_subs, current_proof, show_path, explain_failure, explain_on_fail_only):
                        yield s, p
        if not matched_any and explain_failure and explain_on_fail_only:
            yield None, proof + [("no_unification", functor_to_str(first))]"""
    
    def _solve(
    self, goals, subs, proof,
    show_path=False, explain_failure=False, explain_on_fail_only=False,
    depth=0, max_depth=30, visited=None
):
    
        if visited is None:
            visited = set()
        if depth > max_depth:
            if explain_failure and explain_on_fail_only:
                yield None, proof + [("fail", f"Max recursion depth {max_depth} exceeded")]
            return
        if not goals:
            yield subs, proof
            return
        first, *rest = goals

        # Negation as failure: not(Goal)
        if isinstance(first, Functor) and first.name == 'not' and len(first.args) == 1:
            neg_goal = first.args[0]
            found = False
            for _, _ in self._solve([neg_goal], dict(subs), proof, show_path, explain_failure, explain_on_fail_only, depth=depth+1, max_depth=max_depth, visited=visited):
                found = True
                break
            if not found:
                new_proof = proof + [("not", functor_to_str(neg_goal))]
                yield from self._solve(rest, subs, new_proof, show_path, explain_failure, explain_on_fail_only, depth=depth+1, max_depth=max_depth, visited=visited)
            else:
                if explain_failure and explain_on_fail_only:
                    yield None, proof + [("negation_failed", functor_to_str(neg_goal))]
            return

        # Arithmetic built-ins and Python function assignment
        built_in_subst = try_arithmetic_builtin(first, subs, self)
        if built_in_subst is not None:
            new_proof = proof + [("builtin", functor_to_str(first))]
            yield from self._solve(rest, built_in_subst, new_proof, show_path, explain_failure, explain_on_fail_only, depth=depth+1, max_depth=max_depth, visited=visited)
            return

        if not isinstance(first, Functor):
            if explain_failure and explain_on_fail_only:
                yield None, proof + [("fail_not_functor", functor_to_str(first))]
            return
        key = (first.name, first.arity())
        if key not in self.rules:
            if explain_failure and explain_on_fail_only:
                yield None, proof + [("no_matching_fact_or_rule", functor_to_str(first))]
            return

        # --- Loop/Cycle check (optional) ---
        # Use a tuple (goal string, frozen subs) as signature
        sig = (functor_to_str(first), frozenset(pretty_substitution(subs).items()))
        if sig in visited:
            # Cycle detected: prevent infinite loop
            return
        visited = set(visited)  # copy for this branch
        visited.add(sig)

        for (rule_head, rule_body) in self.rules[key]:
            fresh_head, fresh_body = standardize_apart_pair(rule_head, rule_body)
            new_subs = unify(first, fresh_head, dict(subs))
            if new_subs is not None:
                step = ("rule", functor_to_str(fresh_head), [functor_to_str(b) for b in fresh_body])
                body_goals = [apply_substitution(b, new_subs) for b in fresh_body]
                rest_goals = [apply_substitution(r, new_subs) for r in rest]

                # --- Solve all body goals recursively (cartesian product style) ---
                for s_body, p_body in self._prove_body(body_goals, new_subs, proof + [step], show_path, explain_failure, explain_on_fail_only, depth+1, max_depth, visited):
                    for s_rest, p_rest in self._solve(rest_goals, s_body, p_body, show_path, explain_failure, explain_on_fail_only, depth+1, max_depth, visited):
                        yield s_rest, p_rest

    def _prove_body(
        self, body_goals, subs, proof,
        show_path, explain_failure, explain_on_fail_only, depth, max_depth, visited
    ):
        """Helper: recursively proves a conjunction of body_goals."""
        if not body_goals:
            yield subs, proof
            return
        first, *rest = body_goals
        for s, p in self._solve([first], subs, [], show_path, explain_failure, explain_on_fail_only, depth, max_depth, visited):
            if s is not None:
                for s_rest, p_rest in self._prove_body(rest, s, proof + p, show_path, explain_failure, explain_on_fail_only, depth, max_depth, visited):
                    yield s_rest, p_rest


def pretty_substitution(subs):
    result = {}
    if subs is None:
        return {}
    for var, val in subs.items():
        if isinstance(var, Variable):
            ground_val = apply_substitution(val, subs)
            result[var.name] = str(ground_val)
    return result

# --------------------- Usage Example with Failure Explanation ----------------------



def show_results(results):
    output = [
            ProofResult(
                stat=True if status == "success" else False,
                sub=pretty_substitution(subs) if subs else [],
                proof=proof
            )
            for subs, proof, status in results
        ]
    #print(output)
    """ for subs, proof, status in results:
        if status == "success":
            any_success = True
            #print("SUCCESS:", pretty_substitution(subs))
            if explain_failure:
                print("Proof path:")
                for step in proof:
                    print("   ", step)
                print()
        else:
            any_failure = True
            #print("Failure")
            if explain_failure:
                print("Failure explanation:")
                for step in proof:
                    print("   ", step)
                print() """
    return output[0]



class ProofBundle:
    def __init__(self, stat, sub, proof):
        self.stat = stat
        self.sub = sub
        self.proof = proof

def show_results_1(results):
    """
    - If only one result: returns ProofBundle with scalar values.
    - If multiple: returns ProofBundle with lists for stat/sub/proof.
    """
    output = [
        ProofResult(
            stat=(status == "success"),
            sub=pretty_substitution(subs) if subs else {},
            proof=proof
        )
        for subs, proof, status in results
    ]
    if len(output) == 1:
        # Scalar values
        return ProofBundle(
            output[0].stat,
            output[0].sub,
            output[0].proof
        )
    else:
        # Lists
        return ProofBundle(
            [o.stat for o in output],
            [o.sub for o in output],
            [o.proof for o in output]
        )






def generate_clause(predicate,args):
    head=predicate
    start="("
    for i in args:
        start=start+str(i)+","
    body=start[0:-1]+")"
    clause=head+body
    return clause

def modify_bcrl(P1,Col_Reg_Rules, py_function):
    head_literals = []
    for i in Col_Reg_Rules:
        head= pygol.Meta(i).head
        if head not in head_literals:
            head_literals.append(head)
    for k,v in P1.items():
        #print("Processing:", k)
        kb_bcrl = KnowledgeBase()
        if py_function:
            for fname, f in py_function.items():
                kb_bcrl.register_py_function(fname, f)
        kb_bcrl.add_clauses(Col_Reg_Rules)
        kb_bcrl.add_clauses(v)

        for j in head_literals:
            #print("Querying:", j)

            # Query the knowledge base for the head literal
            results = show_results_1(kb_bcrl.query(j)).sub
            status= show_results_1(kb_bcrl.query(j)).stat
            proof= show_results_1(kb_bcrl.query(j)).proof
            #print("\t",proof)
            
            if status and results:
                if type(results)==dict:
                    results = [results]
                clause_1=""
                #print(results)
                for eachr in results:
                    #print(eachr)
                    args = []
                    for l in pygol.Clause(j).args():
                        #rint(eachr[l])
                        
                        if l in eachr.keys():
                            #if "Var" in eachr[l]:
                                #print("dany", sub[eachr[l]])
                            args.append(eachr[l])
                        else:
                            args.append(l)
                    #print(len(set(args)), len(args))
                    if len(set(args))==len(args):
                        clause_1 = generate_clause(pygol.Clause(j).predicate, args)
                    #print("\t",j, clause_1)
                if clause_1:
                    P1[k].append(clause_1)
    return P1


def write_list_to_file(filename, list_of_strings):
    with open(filename, 'w') as f:
        for item in list_of_strings:
            f.write(str(item) + "." +'\n')


import re

def convert_h(rule_list):
    replaced = []
    for rule in rule_list:
        # Replace (A,B) with (X,Y)
        rule = re.sub(r'\bA\b', 'X', rule)
        rule = re.sub(r'\bB\b', 'Y', rule)
        replaced.append(rule)
    return replaced

def substitue(facts_1,subst):
    args = pygol.Clause(facts_1).args()
    args_1 = []
    if type(subst)== dict:
        subst=[subst]
    for eachs in subst:
        for i in args:
            if pygol.check_argument_type(i)=="variable":
                if eachs[i] not in args_1:
                    args_1.append(eachs[i])
        
    return args_1

def substitue_1(facts_1,subst):
    args = pygol.Clause(facts_1).args()
    clause_set = []
    if type(subst)== dict:
        subst=[subst]
    for eachs in subst:
        args_1 = []
        for i in args:
            if pygol.check_argument_type(i)=="variable":
                if eachs[i] not in args_1:
                    args_1.append(eachs[i])
            else:
                args_1.append(i)
        clause_1 = generate_clause(pygol.Clause(facts_1).predicate, args_1)
        #print(clause_1)
        if clause_1 not in clause_set:
            clause_set.append(clause_1)
    return clause_set


def constrain(P1, constrain={}):
    """
    Filters each list of facts in P1 so that for predicates in `constrain`,
    only the first N occurrences (default: 1) are kept.

    Parameters:
    - P1 (dict): Mapping of target predicate to list of facts.
    - constrain (dict): Predicate name -> number of allowed occurrences.

    Returns:
    - dict: Filtered dictionary.
    """
    filtered = {}

    for key, facts in P1.items():
        seen_count = {pred: 0 for pred in constrain}  # Track per predicate
        new_facts = []

        for fact in facts:
            added = False
            for pred, limit in constrain.items():
                if fact.startswith(pred + "("):
                    if seen_count[pred] < limit:
                        new_facts.append(fact)
                        seen_count[pred] += 1
                    added = True
                    break
            if not added:
                new_facts.append(fact)

        filtered[key] = new_facts
    #print("filtered", filtered)
    return filtered



def evaluate_theory(H, BK,  pos, neg, verbose=False):
    kb_test= pl.KnowledgeBase()
    kb_test.add_clauses(BK)
    kb_test.add_clause("gt(A1,B1):- >(A1,B1)")
    kb_test.add_clause("lt(A1,B1):- <(A1,B1)")
    kb_test.add_clause("range(A2,B2,C2):- >=(A2,B2), =<(A2,C2)")
    kb_test.add_clauses(H)
    pos_count = 0
    neg_count = 0
    for i in pos:
        status= pl.show_results_1(kb_test.query(i)).stat
        #pr=pl.show_results_1(kb_test.query(i)).proof
        #print(pr)
        if status:
            pos_count=pos_count+1

    for i in neg:
        status= pl.show_results_1(kb_test.query(i)).stat
        if not status:
            neg_count=neg_count+1

    if not verbose:
        print("+------------+ Test Results +------------+")
        rec = [['n = ' + str(len(pos) + (len(neg))), 'Positive\n(Actual)', 'Negative\n(Actual)'],
                   ["Positive\n(Predicted)", pos_count, len(neg) - neg_count],
                   ["Negative\n(Predicted)", len(pos) - pos_count, neg_count]]
        table = Texttable()
        table.add_rows(rec)
        print(table.draw())
    cm, accuracy, precision, sensitivity, specificity, fscore = pygol.metrics(pos_count, neg_count, len(neg) - neg_count,
                                                                        len(pos) - pos_count)
    if not verbose:
        rec = [["Metric", "#"], ["Accuracy", accuracy], ["Precision", precision], ["Sensitivity", sensitivity],
                   ["Specificity", specificity], ["F1 Score", fscore]]
        table = Texttable()
        table.add_rows(rec)
        print(table.draw())
        return pygol.PyGol([], cm,  accuracy, precision, sensitivity, specificity, fscore, [])
    else:
        return pygol.PyGol([], [], 0, 0, 0, 0, 0, [])

