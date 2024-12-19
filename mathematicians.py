# here we define a society of mathematicians that work together to solve problems from math dataset. 
# First there is a decomposer that decomposes the problem into smaller subproblems and distributes them to the mathematicians. 
# Then there are mathemaricians that specialize in different feilds like algebra, calculus, geometry, etc. 
# they work with other math researchers that are further specialized in different areas like number area calculation, counting principal,etc.
# after conversation, they report their answers to an aggregator that aggregates the answers and returns the final answer.
# Currently we start with MATH dataset, then we include HardMAth, Scibench etc as well. 

'''
Instantiate agents
make few graphs
load datasets
evaluate those graphs on dataset
'''
from society import Agent, Society
from llm import *
import json
from tqdm import tqdm


from datasets import load_dataset

ds = load_dataset("ai2lumos/lumos_maths_plan_iterative")

# provide 4 shots of examples of problem solving to each agent related to their topic.
geometry_agent = {
    "name": "geometry",
    "id": 1,
    "system_prompt": '''You are a highly skilled mathematical assistant specializing in geometry. Your task is to solve geometry problems with precision and clarity. You understand concepts such as Euclidean geometry, coordinate geometry, trigonometry, and geometric proofs.
    
    You are given 4 examples of how to solve geometry problems. You can use these examples to solve the problems.

    Question: A plane intersects a right circular cylinder of radius $1$ forming an ellipse. If the major axis of the ellipse of $50%$ longer than the minor axis, the length of the major axis is  $textbf{(A)} 1qquad textbf{(B)} frac{3}{2}qquad textbf{(C)}2qquad textbf{(D)} frac{9}{4}qquad textbf{(E)} 3$

    Answer: The minor axis is $2(1) = 2$. Since the major axis is $50%$ longer, its length is $2(1.5) = [Answer: 3]$.

    Question: Let $triangle ABC$ be an acute scalene triangle with circumcircle $omega$. The tangents to $omega$ at $B$ and $C$ intersect at $T$. Let $X$ and $Y$ be the projections of $T$ onto lines $AB$ and $AC$, respectively. Suppose $BT = CT = 16$, $BC = 22$, and $TX^2 + TY^2 + XY^2 = 1143$. Find $XY^2$.

    Answer: Using geometry and algebra, $XY^2$ is solved as $[Answer: 717]$.


    Question: The consecutive angles of a particular trapezoid form an arithmetic sequence. If the largest angle measures $120^{circ}$, what is the measure of the smallest angle?  

    Answer: The smallest angle is $180^circ - 120^circ = [Answer: 60]$.

    Question: Square ABCD has its center at $(8,-8)$ and has an area of 4 square units. The top side of the square is horizontal. The square is then dilated with the dilation center at (0,0) and a scale factor of 2. What are the coordinates of the vertex of the image of square ABCD that is farthest from the origin? Give your answer as an ordered pair.

    Answer: After dilation, the farthest point from the origin is $[Answer: (18, -18)]$.

    When solving problems, provide detailed yet concise answers with examples. Always maintain a logical and structured approach to problem-solving. It could be that the problem is not a geometry problem, in that case, you should say that it is out of your domain. Think about the solution and Give the final answer in brackets like [Answer: <answer>] where <answer> is a numeric value like [Answer: 123] or [Answer: 123.45] etc. 
    ''',
    "model_name": "gemma_base"
    }

permutation_agent = {
    "name": "permutation",
    "id": 2,
    "system_prompt": '''
    You are a highly skilled mathematical assistant specializing in combinatorics. Your task is to solve problems involving permutations, combinations, counting principles with precision and clarity.

    You are given 4 examples of how to solve geometry problems. You can use these examples to solve the problems.

    Question: What is the value of $9^3 + 3(9^2) + 3(9) + 1$?

    Answer: This is $(1 + 9)^3 = 10^3 = [Answer: 1000]$.

    Question: Two different integers are randomly chosen from the set $${ -5, -8, 7, 4, -2 }.$$ What is the probability that their product is negative? Express your answer as a common fraction.

    Answer: There are $6$ ways to choose one positive and one negative integer, and $10$ total ways to choose any two integers. So the probability is $frac{6}{10} = [Answer: frac{3}{5}]$.

    Question: Bob's password consists of a non-negative single-digit number followed by a letter and another non-negative single-digit number (which could be the same as the first one). What is the probability that Bob’s password consists of an odd single-digit number followed by a letter and a positive single-digit number?

    Answer: The probability is $frac{1}{2} cdot 1 cdot frac{9}{10} = [Answer: frac{9}{20}]$.

    Question: How many different positive, four-digit integers can be formed using the digits 2, 2, 9, and 9?

    Answer: Accounting for repeated digits, the number of integers is $frac{4!}{2! cdot 2!} = [Answer: 6]$.

    When solving problems, provide detailed yet concise answers with relevant examples. Always maintain a logical and structured approach to problem-solving. It could be that the problem is not a combinatorics problem, in that case, you should say that it is out of your domain. Think about the solution and Give the final answer in brackets like [Answer: <answer>] where <answer> is a numeric value like [Answer: 123] or [Answer: 123.45] etc.''',
    "model_name": "gemma_base"
    }

number_theory_agent = {
    "name": "number_theory",
    "id": 3,
    "system_prompt": '''You are a highly skilled mathematical assistant specializing in number theory. Your task is to solve problems related to prime numbers, divisibility, modular arithmetic, greatest common divisors, least common multiples, Diophantine equations, and more with precision and clarity.

    You are given 4 examples of how to solve geometry problems. You can use these examples to solve the problems.

    Question: Find the remainder when the sum [75+76+77+78+79+80+81+82] is divided by 16.
    Answer: The sum simplifies modulo 16 to $75 + 76 + 77 pmod\\{16\\} = -5 -4 -3 pmod{16} equiv -12 equiv 4 pmod{16}$. [Answer: 4]

    Question: The prime factorization of 2160 is $2^4 times 3^3 times 5$. How many of its positive integer factors are perfect squares?
    Answer: Perfect square factors require even exponents: $3$ choices for $a$ and $2$ for $b$, yielding $3 cdot 2 = [Answer: 6]$.

    Question: How many positive divisors of $150$ are not divisible by 5?
    Answer: Excluding $p=1$ for $5^p$, there are $2 cdot 2 = [Answer: 4]$ divisors satisfying this condition.

    Question: What is the greatest common divisor of $5!$ and $frac{8!}{3!}$?
    Answer: Lowest powers of common primes give $2^3 cdot 3 cdot 5 = [Answer: 120]$.

    When solving problems provide detailed yet concise answers with relevant examples. Always maintain a structured and methodical approach to problem-solving. It could be that the problem is not a number theory problem, in that case, you should say that it is out of your domain. Give the final answer in brackets like [Answer: <answer>] where <answer> is a numeric value like [Answer: 123] or [Answer: 123.45] etc''',
    "model_name": "gemma_base"
    }

algebra_agent = {
    "name": "algebra",
    "id": 4,
    "system_prompt": '''You are a highly skilled mathematical assistant specializing in algebra. Your task is to solve problems involving equations, inequalities, polynomials, matrices, and linear algebra with precision and clarity.

    You are given 4 examples of how to solve geometry problems. You can use these examples to solve the problems.

    Question: Let $p$ and $q$ be the two distinct solutions to the equation $$frac{4x-12}{x^2+2x-15}=x+2.$$If $p > q$, what is the value of $p - q$?
    Answer: Simplifying, the equation becomes ( frac{4}{x+5} = x+2 ). Solving yields ( x^2 + 7x + 6 = 0 ), which factors to ( (x+6)(x+1) = 0 ). Thus, ( p = -1 ), ( q = -6 ), and ( p - q = [Answer: 5] ).  

    Question: What is the largest value of $x$ such that the expression [dfrac{x+1}{8x^2-65x+8}] is not defined?
    Answer: The denominator is zero when ( 8x^2 - 65x + 8 = 0 ). Factoring gives ( (8x-1)(x-8) = 0 ), so ( x = frac{1}{8} ) or ( x = 8 ). The largest value is [Answer: 8].  

    Question: What is the value of ( 525^2 - 475^2 )?  
    Answer: Using the difference of squares formula, ( 525^2 - 475^2 = (525+475)(525-475) = 1000 cdot 50 = [Answer: 50000] ).  

    Question: What is the largest three-digit integer whose digits are distinct and form a geometric sequence?  
    Answer: For a geometric sequence with hundreds digit ( 9 ), using a common ratio of ( r = frac{3}{2} ), the number is ( 964 ). Hence, the largest number is [Answer: 964].  

    When solving problems provide detailed yet concise answers with relevant examples. Always maintain a logical and systematic approach to problem-solving. It could be that the problem is not a algebra problem, in that case, you should say that it is out of your domain. Think about the solution and Give the final answer in brackets like [Answer: <answer>] where <answer> is a numeric value like [Answer: 123] or [Answer: 123.45] etc.''',
    "model_name": "gemma_base"
    }

calculus_agent = {
    "name": "calculus",
    "id": 5,
    "system_prompt": '''You are a highly skilled mathematical assistant specializing in calculus. Your task is to solve problems involving differentiation, integration, limits, series, and multivariable calculus with precision and clarity.
    
    You are given 4 examples of how to solve geometry problems. You can use these examples to solve the problems.

    Question: Find the number of solutions ton[cos 4x + cos^2 3x + cos^3 2x + cos^4 x = 0]nfor (-pi le x le pi.)
    Answer: Rewriting all terms in terms of ( cos 2x ), the equation simplifies ton[12 cos^3 2x + 9 cos^2 2x - 4 cos 2x - 1 = 0.]nFactoring, we getn[(cos 2x + 1)(12 cos^2 2x - 3 cos 2x - 1) = 0.]nFor ( cos 2x + 1 = 0 ), ( cos 2x = -1 ), which gives 2 solutions (( pm frac\\{pi\\}{2} )).nFor ( 12 cos^2 2x - 3 cos 2x - 1 = 0 ), using the quadratic formula,n[cos 2x = frac{3 pm sqrt{57}}{12}.]nBoth roots lie in ([-1, 1]), yielding 4 solutions each. The total number of solutions isn[2 + 4 + 4 = [Answer: 10].]nn
    
    Question: Let ( S ) be a region in the plane with area 10. When we apply the matrixn[begin{pmatrix} 2 & 1  7 & -3 end{pmatrix}]nto ( S ), we obtain the region ( S' ). Find the area of ( S' ).
    Answer: The determinant of the matrix isn[begin{vmatrix} 2 & 1  7 & -3 end{vmatrix} = (2)(-3) - (1)(7) = -13.]nThe scaling factor for the area is ( |-13| = 13 ). Thus, the area of ( S' ) isn[13 cdot 10 = [Answer: 130].]nn
    
    Question: Find the area of the triangle with vertices ((-1,4)), ((7,0)), and ((11,5)).
    Answer: Using vectors,n[mathbf{v} = overrightarrow{CA} = begin{pmatrix} -12  -1 end{pmatrix}, quad  nmathbf{w} = overrightarrow{CB} = begin{pmatrix} -4  -5 end{pmatrix}.]nThe area of the parallelogram spanned by ( mathbf{v} ) and ( mathbf{w} ) isn[|(-12)(-5) - (-4)(-1)| = 56.]nThe area of the triangle is half of this,n[frac{56}{2} = [Answer: 28].]
        
    Question: The sumn[sum_{x=2}^{44} 2sin x sin 1 [1 + sec (x-1) sec (x+1)]]ncan be written in the formn[sum_{n=1}^{4} (-1)^n frac{Phi(theta_n)}{Psi(theta_n)},]nwhere ( Phi, Psi ) are trigonometric functions and ( theta_1, theta_2, theta_3, theta_4 ) are degrees in ([0,45]). Find ( theta_1 + theta_2 + theta_3 + theta_4 ).
    Answer: Using the product-to-sum identities, the sum simplifies and telescopes ton[-frac{sin^2(1)}{cos(1)} - frac{sin^2(2)}{cos(2)} + frac{sin^2(44)}{cos(44)} + frac{sin^2(45)}{cos(45)}.]nThe four terms correspond to ( theta_1 = 1^circ ), ( theta_2 = 2^circ ), ( theta_3 = 44^circ ), ( theta_4 = 45^circ ). Summing these,n[theta_1 + theta_2 + theta_3 + theta_4 = 1 + 2 + 44 + 45 = [Answer: 92].

    When solving problems provide detailed yet concise answers with relevant examples and visual aids if needed. Always maintain a logical and structured approach to problem-solving. It could be that the problem is not a calculus problem, in that case, you should say that it is out of your domain. Think about the solution and Give the final answer in brackets like [Answer: <answer>] where <answer> is a numeric value like [Answer: 123] or [Answer: 123.45] etc.
    ''',
    "model_name": "gemma_base"
    }

aggregator_agent = {
    "name": "aggregator",
    "id": 6,
    "system_prompt": '''You are an aggregator agent tasked with synthesizing and evaluating responses from specialized mathematical agents (Number Theory Agent, Calculus Agent, Algebra Agent, Geometry Agent, and Combinatorics Agent). Your goal is to carefully analyze their responses along with the original question and provide the most accurate and well-justified answer.

    Present the final answer:
    1. Clearly and concisely, with logical reasoning and justification for your decision.
    2. Ensure the solution is correct, easy to understand, and formatted using proper mathematical notation.

    Give the final answer in brackets like [Answer: <answer>] where <answer> is a numeric value like [Answer: 123] or [Answer: 123.45] etc.
    ''',
    "model_name": "gemma_base"
    }

def test_mathematicians(llm_name, question_list, gold_answer_list, problem_types, log_file_name):
    acc = 0
    question_list = question_list[:100]
    gold_answer_list = gold_answer_list[:100]
    problem_types = problem_types[:100]
    for i in tqdm(range(len(question_list))):
        
        question = question_list[i]
        gold_answer = gold_answer_list[i]
        problem_type = problem_types[i]
        geometry = Agent(**geometry_agent)
        permutation = Agent(**permutation_agent)
        number_theory = Agent(**number_theory_agent)
        algebra = Agent(**algebra_agent)
        calculus = Agent(**calculus_agent)
        aggregator = Agent(**aggregator_agent)


        mathematicians = [geometry, permutation, number_theory, algebra, calculus, aggregator]

        heirarchical_mathematicians_graph = {
            "aggregator": [],
            "geometry": [aggregator],
            "permutation": [aggregator],
            "number_theory": [aggregator],
            "algebra": [aggregator],
            "calculus": [aggregator],
        }

        clique_mathematicians_graph = {
            "aggregator": [],
            "geometry": [permutation, number_theory, algebra, calculus, aggregator],
            "permutation": [geometry, number_theory, algebra, calculus, aggregator],
            "number_theory": [geometry, permutation, algebra, calculus, aggregator],
            "algebra": [geometry, permutation, number_theory, calculus, aggregator],
            "calculus": [geometry, permutation, number_theory, algebra, aggregator],
        }
        
        if("algebra" in problem_type[0].lower()):
            problem_specific_agent = algebra
        elif ("probability" in problem_type[0].lower()):
            problem_specific_agent = permutation
        elif ("gemoetry" in problem_type[0].lower()):
            problem_specific_agent = geometry
        elif ("number" in problem_type[0].lower()):
            problem_specific_agent = number_theory
        elif ("calculus" in problem_type[0].lower()):
            problem_specific_agent = calculus
        
        single_mathematician = [problem_specific_agent, aggregator]
        single_agent_graph = {
            "aggregator": [],
            "geometry": [aggregator],
            "permutation": [aggregator],
            "number_theory": [aggregator],
            "algebra": [aggregator],
            "calculus": [aggregator],
        }

        # instantiate the society
        # ProblemSolver = Society(mathematicians, heirarchical_mathematicians_graph)
        # ProblemSolver = Society(mathematicians, clique_mathematicians_graph)
        ProblemSolver = Society(single_mathematician, single_agent_graph)
        ProblemSolver.run_simulation(1, question, background = "")

        # based on the memory of head doctor, give the final answer to the user
        # breakpoint()
        # aggregator_response = aggregator.last_utterance + question
        
        # messages = [
        #     {"role": "system", "content": answer_extraction_prompt},
        #     {"role": "user", "content": aggregator_response + "n Answer: Let's think step by step."}
        # ]

        # response = LLM(messages, llm_name)
        response = aggregator.last_utterance
        # print(response)
        try:
            response = response.split("[Answer:")[1].strip()
        except:
            response = response.strip()
        # print(response)
        # print(response, gold_answer)
        if(gold_answer in response):
            acc+=1
        # print(acc)

        # save the question, answer and memory of each agent in a jsonl file
        jsonl_content = {
            "question": question, 
            "gold_answer": gold_answer, 
            "society_answer":response, 
            "problem_type": problem_type,
            "geometry": geometry.memory, 
            "permutation": permutation.memory, 
            "number_theory": number_theory.memory, 
            "algebra": algebra.memory, 
            "calculus": calculus.memory, 
            "aggregator": aggregator.memory
        }
        with open(log_file_name, "a") as f:
            f.write(json.dumps(jsonl_content) + "\n")

    print("Accuracy: ", acc/len(question_list))







