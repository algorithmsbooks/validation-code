"""
This document is an automatically-generated file that contains all typeset code blocks from
Algorithms for Validation by Mykel J. Kochenderfer, Sydney M. Katz, Anthony L. Corso, and Robert J. Moss. A PDF version of the book is available online at algorithmsbook.com/validation.

We share this content in the hopes that it helps you and makes the validation algorithms
more approachable and accessible. Thank you for reading!

If you encounter any issues or have pressing comments, please file an issue at
github.com/algorithmsbooks/validation.
"""

#################### introduction 1
abstract type Agent end
abstract type Environment end
abstract type Sensor end

struct System
    agent::Agent
    env::Environment
    sensor::Sensor
end
####################

#################### introduction 2
function step(sys::System, s)
    o = sys.sensor(s)
    a = sys.agent(o)
    s′ = sys.env(s, a)
    return (; o, a, s′)
end

function rollout(sys::System; d)
    s = rand(Ps(sys.env))
    τ = []
    for t in 1:d
        o, a, s′ = step(sys, s)
        push!(τ, (; s, o, a))
        s = s′
    end
    return τ
end
####################

#################### introduction 3
abstract type Specification end
function evaluate(ψ::Specification, τ) end
isfailure(ψ::Specification, τ) = !evaluate(ψ, τ)
####################

#################### model_building 1
struct MaximumLikelihoodParameterEstimation
    likelihood # p(y) = likelihood(x; θ)
    optimizer  # optimization algorithm: θ = optimizer(f)
end

function fit(alg::MaximumLikelihoodParameterEstimation, data)
    f(θ) = sum(-logpdf(alg.likelihood(x, θ), y) for (x,y) in data)
    return alg.optimizer(f)
end
####################

#################### model_building 2
struct BayesianParameterEstimation
    likelihood # p(y) = likelihood(x, θ)
    prior      # prior distribution
    sampler    # Turing.jl sampler
    m          # number of samples from posterior
end


function fit(alg::BayesianParameterEstimation, data)
    x, y = first.(data), last.(data)
    @model function posterior(x, y)
        θ ~ alg.prior
        for i in eachindex(x)
            y[i] ~ alg.likelihood(x[i], θ)
        end
    end
    return Turing.sample(posterior(x, y), alg.sampler, alg.m)
end
####################

#################### property_specification 1
struct LTLSpecification <: Specification
	formula # formula specified using SignalTemporalLogic.jl
end
evaluate(ψ::LTLSpecification, τ) = ψ.formula([step.s for step in τ])
####################

#################### property_specification 2
struct STLSpecification <: Specification
	formula # formula specified using SignalTemporalLogic.jl
    I       # time interval (e.g. 3:10)
end
evaluate(ψ::STLSpecification, τ) = ψ.formula([step.s for step in τ[ψ.I]])
####################

#################### falsification 1
struct DirectFalsification
    d # depth
    m # number of samples
end

function falsify(alg::DirectFalsification, sys, ψ)
    d, m = alg.d, alg.m
    τs = [rollout(sys, d=d) for i in 1:m]
    return filter(τ->isfailure(ψ, τ), τs)
end
####################

#################### falsification 2
struct Disturbance
    xa # agent disturbance
    xs # environment disturbance
    xo # sensor disturbance
end

struct DisturbanceDistribution
    Da # agent disturbance distribution
    Ds # environment disturbance distribution
    Do # sensor disturbance distribution
end

function step(sys::System, s, D::DisturbanceDistribution)
    xo = rand(D.Do(s))
    o = sys.sensor(s, xo)
    xa = rand(D.Da(o))
    a = sys.agent(o, xa)
    xs = rand(D.Ds(s, a))
    s′ = sys.env(s, a, xs)
    x = Disturbance(xa, xs, xo)
    return (; o, a, s′, x)
end
####################

#################### falsification 3
abstract type TrajectoryDistribution end
function initial_state_distribution(p::TrajectoryDistribution) end
function disturbance_distribution(p::TrajectoryDistribution, t) end
function depth(p::TrajectoryDistribution) end
####################

#################### falsification 4
struct NominalTrajectoryDistribution <: TrajectoryDistribution
    Ps # initial state distribution
    D  # disturbance distribution
    d  # depth
end

function NominalTrajectoryDistribution(sys::System, d)
	D = DisturbanceDistribution((o) -> Da(sys.agent, o),
								(s, a) -> Ds(sys.env, s, a),
								(s) -> Do(sys.sensor, s))
	return NominalTrajectoryDistribution(Ps(sys.env), D, d)
end

initial_state_distribution(p::NominalTrajectoryDistribution) = p.Ps
disturbance_distribution(p::NominalTrajectoryDistribution, t) = p.D
depth(p::NominalTrajectoryDistribution) = p.d
####################

#################### falsification 5
function rollout(sys::System, p::TrajectoryDistribution; d=depth(p))
    s = rand(initial_state_distribution(p))
    τ = []
    for t in 1:d
        o, a, s′, x = step(sys, s, disturbance_distribution(p, t))
        push!(τ, (; s, o, a, x))
        s = s′
    end
    return τ
end
####################

#################### falsification 6
function step(sys::System, s, x)
    o = sys.sensor(s, x.xo)
    a = sys.agent(o, x.xa)
    s′ = sys.env(s, a, x.xs)
    return (; o, a, s′)
end

function rollout(sys::System, s, 𝐱; d=length(𝐱))
    τ = []
    for t in 1:d
        x = 𝐱[t]
        o, a, s′ = step(sys, s, x)
        push!(τ, (; s, o, a, x))
        s = s′
    end
    return τ
end
####################

#################### falsification 7
function robustness_objective(x, sys, ψ; smoothness=0.0)
    s, 𝐱 = extract(sys.env, x)
    τ = rollout(sys, s, 𝐱)
    𝐬 = [step.s for step in τ]
    return robustness(𝐬, ψ.formula, w=smoothness)
end
####################

#################### falsification 8
function Distributions.logpdf(D::DisturbanceDistribution, s, o, a, x)
    logp_xa = logpdf(D.Da(o), x.xa)
    logp_xs = logpdf(D.Ds(s, a), x.xs)
    logp_xo = logpdf(D.Do(s), x.xo)
    return logp_xa + logp_xs + logp_xo
end

function Distributions.pdf(p::TrajectoryDistribution, τ)
    logprob = logpdf(initial_state_distribution(p), τ[1].s)
    for (t, step) in enumerate(τ)
        s, o, a, x = step
        logprob += logpdf(disturbance_distribution(p, t), s, o, a, x)
    end
    return exp(logprob)
end
####################

#################### falsification 9
function likelihood_objective(x, sys, ψ; smoothness=0.0)
    s, 𝐱 = extract(sys.env, x)
    τ = rollout(sys, s, 𝐱)
    if isfailure(ψ, τ)
        p = NominalTrajectoryDistribution(sys, length(𝐱))
        return -pdf(p, τ)
    else
        𝐬 = [step.s for step in τ]
        return robustness(𝐬, ψ.formula, w=smoothness)
    end
end
####################

#################### falsification 10
function weighted_likelihood_objective(x, sys, ψ; smoothness=0.0, λ=1.0)
    s, 𝐱 = extract(sys.env, x)
    τ = rollout(sys, s, 𝐱)
    𝐬 = [step.s for step in τ]
    p = NominalTrajectoryDistribution(sys, length(𝐱))
    return robustness(𝐬, ψ.formula, w=smoothness) - λ * log(pdf(p, τ))
end
####################

#################### falsification 11
struct OptimizationBasedFalsification
    objective # objective function
    optimizer # optimization algorithm
end

function falsify(alg::OptimizationBasedFalsification, sys, ψ)
    f(x) = alg.objective(x, sys, ψ)
    return alg.optimizer(f, sys, ψ)
end
####################

#################### planning 1
defect(τᵢ, τᵢ₊₁) = norm(τᵢ₊₁[1].s - τᵢ[end].s)

function shooting_robustness(x, sys, ψ; smoothness=0.0, λ=1.0)
    segments = extract(sys.env, x)
    n = length(segments)
    τ_segments = [rollout(sys, seg.s, seg.𝐱) for seg in segments]
    τ = vcat(τ_segments...)
    𝐬 = [step.s for step in τ]
    ρ = smooth_robustness(𝐬, ψ.formula, w=smoothness)
    defects = [defect(τ_segments[i], τ_segments[i+1]) for i in 1:n-1]
    return ρ + λ*sum(defects)
end
####################

#################### planning 2
abstract type TreeSearch end

function falsify(alg::TreeSearch, sys, ψ)
    tree = initialize_tree(alg, sys)
    for i in 1:alg.k_max
        node = select(alg, sys, ψ, tree)
        extend!(alg, sys, ψ, tree, node)
    end
    return failures(tree, sys, ψ)
end
####################

#################### planning 3
function trajectory(node)
    τ = []
    while !isnothing(node.parent)
        pushfirst!(τ, (s=node.parent.state, node.edge...))
        node = node.parent
    end
    return τ
end

function failures(tree, sys, ψ)
    leaves = filter(node -> isempty(node.children), tree)
    τs = [trajectory(node) for node in leaves]
    return filter(τ -> isfailure(ψ, τ), τs)
end
####################

#################### planning 4
struct RRT <: TreeSearch
    sample_goal        # sgoal = sample_goal(tree)
    compute_objectives # objectives = compute_objectives(tree, sgoal)
    select_disturbance # x = select_disturbance(sys, node)
    k_max              # number of iterations
end

mutable struct RRTNode
    state       # node state
    parent      # parent node
    edge        # (o, a, x)
    children    # vector of child nodes
    goal_state  # current goal state
end

function initialize_tree(alg::RRT, sys)
    return [RRTNode(rand(Ps(sys.env)), nothing, nothing, [], nothing)]
end

function select(alg::RRT, sys, ψ, tree)
    sgoal = alg.sample_goal(tree)
    objectives = alg.compute_objectives(tree, sgoal)
    node = tree[argmin(objectives)]
    node.goal_state = sgoal
    return node
end

function extend!(alg::RRT, sys, ψ, tree, node)
    x = alg.select_disturbance(sys, node)
    o, a, s′ = step(sys, node.state, x)
    snew = RRTNode(s′, node, (; o, a, x), [], nothing)
    push!(node.children, snew)
    push!(tree, snew)
end
####################

#################### planning 5
random_goal(tree, lo, hi) = rand.(Distributions.Uniform.(lo, hi))

function distance_objectives(tree, sgoal) 
    return [norm(sgoal .- node.state) for node in tree]
end

function random_disturbance(sys, node)
    D = DisturbanceDistribution(sys)
    o, a, s′, x = step(sys, node.state, D)
    return x
end
####################

#################### planning 6
function goal_disturbance(sys, node; m=10)
    D = DisturbanceDistribution(sys)
    steps = [step(sys, node.state, D) for i in 1:m]
    distances = [norm(node.goal_state - step.s′) for step in steps]
    return steps[argmin(distances)].x
end
####################

#################### planning 7
function average_dispersion(points, lo, hi, lengths)
    points_norm = [(point .- lo) ./ (hi .- lo) for point in points]
    ranges = [range(0, 1, length) for length in lengths]
    δ = minimum(Float64(r.step) for r in ranges)
    grid_dispersions = []
    for grid_point in Iterators.product(ranges...)
        dmin = minimum(norm(grid_point .- p) for p in points_norm)
        push!(grid_dispersions, min(dmin, δ) / δ)
    end
    return mean(grid_dispersions)
end
####################

#################### planning 8
function star_discrepancy(points, lo, hi, lengths)
    n, dim = length(points), length(lo)
    𝒱 = [(point .- lo) ./ (hi .- lo) for point in points]
    ranges = [range(0, 1, length)[1:end-1] for length in lengths]
    steps = [Float64(r.step) for r in ranges]
    ℬ = Hyperrectangle(low=zeros(dim), high=ones(dim))
    lbs, ubs = [], []
    for grid_point in Iterators.product(ranges...)
        h⁻ = Hyperrectangle(low=zeros(dim), high=[grid_point...])
        h⁺ = Hyperrectangle(low=zeros(dim), high=grid_point .+ steps)
        𝒱h⁻ = length(filter(v -> v ∈ h⁻, 𝒱))
        𝒱h⁺ = length(filter(v -> v ∈ h⁺, 𝒱))
        push!(lbs, max(abs(𝒱h⁻ / n - volume(h⁻) / volume(ℬ)),
                        abs(𝒱h⁺ / n - volume(h⁺) / volume(ℬ))))
        push!(ubs, max(𝒱h⁺ / n - volume(h⁻) / volume(ℬ),
                        volume(h⁺) / volume(ℬ) - 𝒱h⁻ / n))
    end
    return maximum(lbs), maximum(ubs)
end
####################

#################### planning 9
distance_c(node) = norm(node.parent.state .- node.state)
distance_h(node, sgoal) = norm(sgoal .- node.state)

function cost_objectives(tree, sgoal; c=distance_c, h=distance_h)
    costs = Dict()
    queue = [tree[1]]
    while !isempty(queue)
        node = popfirst!(queue)
        if isnothing(node.parent)
            costs[node] = 0.0
        else
            costs[node] = c(node) + costs[node.parent]
        end
        for child in node.children
            push!(queue, child)
        end
    end
    heuristics = [h(sgoal, node) for node in tree]
    objectives = [costs[node] for node in tree] .+ heuristics
    return objectives
end
####################

#################### planning 10
struct MCTS <: TreeSearch
    estimate_value     # v = estimate_value(sys, ψ, node)
    c                  # exploration constant
    k                  # progressive widening constant
    α                  # progressive widening exponent
    select_disturbance # x = select_disturbance(sys, node)
    k_max              # number of iterations
end

mutable struct MCTSNode
    state    # node state
    parent   # parent node
    edge     # (o, a, x)
    children # vector of child nodes
    N        # visit count
    Q        # value estimate
end

function initialize_tree(alg::MCTS, sys)
    return [MCTSNode(rand(Ps(sys.env)), nothing, nothing, [], 1, 0)]
end

function select(alg::MCTS, sys, ψ, tree)
    c, k, α, node = alg.c, alg.k, alg.α, tree[1]
    while length(node.children) > k * node.N^α
        node = lcb(node, c)
    end
    return node
end

function extend!(alg::MCTS, sys, ψ, tree, node)
    x = alg.select_disturbance(sys, node)
    o, a, s′ = step(sys, node.state, x)
    Q = alg.estimate_value(sys, ψ, s′)
    snew = MCTSNode(s′, node, (; o, a, x), [], 1, Q)
    push!(node.children, snew)
    push!(tree, snew)
    while !isnothing(node)
        node.N += 1
        node.Q += (Q - node.Q) / node.N
        Q, node = node.Q, node.parent
    end
end
####################

#################### planning 11
function lcb(node::MCTSNode, c)
    Qs = [node.Q for node in node.children]
    Ns = [node.N for node in node.children]
    lcbs = [Q - c*sqrt(log(node.N)/N) for (Q, N) in zip(Qs, Ns)]
    return node.children[argmin(lcbs)]
end
####################

#################### failure_distribution 1
struct RejectionSampling
    p̄     # target density
    q     # proposal trajectory distribution
    c     # constant such that p̄(τ) ≤ cq(τ)
    k_max # max iterations
end

function sample_failures(alg::RejectionSampling, sys, ψ)
    p̄, q, c, k_max = alg.p̄, alg.q, alg.c, alg.k_max
    τs = []
    for k in 1:k_max
        τ = rollout(sys, q)
        if rand() < p̄(τ) / (c * pdf(q, τ))
            push!(τs, τ)
        end
    end
    return τs
end
####################

#################### failure_distribution 2
struct MCMCSampling
    p̄        # target density
    g        # kernel: τ′ = rollout(sys, g(τ))
    τ        # initial trajectory
    k_max    # max iterations
    m_burnin # number of samples to discard from burn-in
    m_skip   # number of samples to skip for thinning
end

function sample_failures(alg::MCMCSampling, sys, ψ)
    p̄, g, τ = alg.p̄, alg.g, alg.τ
    k_max, m_burnin, m_skip = alg.k_max, alg.m_burnin, alg.m_skip
    τs = []
    for k in 1:k_max
        τ′ = rollout(sys, g(τ))
        if rand() < (p̄(τ′) * pdf(g(τ′), τ)) / (p̄(τ) * pdf(g(τ), τ′))
            τ = τ′
        end
        push!(τs, τ)
    end
    return τs[m_burnin:m_skip:end]
end
####################

#################### failure_distribution 3
struct ProbabilisticProgramming
    Δ        # distance function: Δ(𝐬)
    mcmc_alg # e.g. Turing.NUTS()
    k_max    # number of samples
    d        # trajectory depth
    ϵ        # smoothing parameter
end

function sample_failures(alg::ProbabilisticProgramming, sys, ψ)
    Δ, mcmc_alg = alg.Δ, alg.mcmc_alg
    k_max, d, ϵ = alg.k_max, alg.d, alg.ϵ

    @model function rollout(sys, d; xo=fill(missing, d),
                                    xa=fill(missing, d),
                                    xs=fill(missing, d))
        p = NominalTrajectoryDistribution(sys, d)
        s ~ initial_state_distribution(p)
        𝐬 = [s, [zeros(length(s)) for i in 1:d]...]
        for t in 1:d
            D = disturbance_distribution(p, t)
            s = 𝐬[t]
            xo[t] ~ D.Do(s)
            o = sys.sensor(s, xo[t])
            xa[t] ~ D.Da(o)
            a = sys.agent(o, xa[t])
            xs[t] ~ D.Ds(s, a)
            𝐬[t+1] = sys.env(s, a, xs[t])
        end
        Turing.@addlogprob! logpdf(Normal(0.0, ϵ), Δ(𝐬))
    end

    return Turing.sample(rollout(sys, d), mcmc_alg, k_max)
end
####################

#################### failure_probability 1
struct DirectEstimation
    d # depth
    m # number of samples
end

function estimate(alg::DirectEstimation, sys, ψ)
    d, m = alg.d, alg.m
    τs = [rollout(sys, d=d) for i in 1:m]
    return mean(isfailure(ψ, τ) for τ in τs)
end
####################

#################### failure_probability 2
struct BayesianEstimation
    prior::Beta # from Distributions.jl
    d           # depth
    m           # number of samples
end

function estimate(alg::BayesianEstimation, sys, ψ)
    prior, d, m = alg.prior, alg.d, alg.m
    τs = [rollout(sys, d=d) for i in 1:m]
    n, m = sum(isfailure(ψ, τ) for τ in τs), length(τs)
    return Beta(prior.α + n, prior.β + m - n)
end
####################

#################### failure_probability 3
struct ImportanceSamplingEstimation
    p # nominal distribution
    q # proposal distribution
    m # number of samples
end

function estimate(alg::ImportanceSamplingEstimation, sys, ψ)
    p, q, m = alg.p, alg.q, alg.m
    τs = [rollout(sys, q) for i in 1:m]
    ps = [pdf(p, τ) for τ in τs]
    qs = [pdf(q, τ) for τ in τs]
    ws = ps ./ qs
    return mean(w * isfailure(ψ, τ) for (w, τ) in zip(ws, τs))
end
####################

#################### failure_probability 4
struct MultipleImportanceSamplingEstimation
    p         # nominal distribution
    qs        # proposal distributions
    weighting # weighting scheme: ws = weighting(p, qs, τs)
end

smis(p, qs, τs) = [pdf(p, τ) / pdf(q, τ) for (q, τ) in zip(qs, τs)]
dmmis(p, qs, τs) = [pdf(p, τ) / mean(pdf(q, τ) for q in qs) for τ in τs]

function estimate(alg::MultipleImportanceSamplingEstimation, sys, ψ)
    p, qs, weighting = alg.p, alg.qs, alg.weighting
    τs = [rollout(sys, q) for q in qs]
    ws = weighting(p, qs, τs)
    return mean(w * isfailure(ψ, τ) for (w, τ) in zip(ws, τs))
end
####################

#################### failure_probability 5
struct CrossEntropyEstimation
    p       # nominal trajectory distribution
    q₀      # initial proposal distribution
    f       # objective function f(τ, ψ)
    k_max   # number of iterations
    m       # number of samples per iteration
    m_elite # number of elite samples
end

function estimate(alg::CrossEntropyEstimation, sys, ψ)
    k_max, m, m_elite = alg.k_max, alg.m, alg.m_elite
    p, q, f = alg.p, alg.q₀, alg.f
    for k in 1:k_max
        τs = [rollout(sys, q) for i in 1:m]
        Y = [f(τ, ψ) for τ in τs]
        order = sortperm(Y)
        γ = max(0, Y[order[m_elite]])
        ps = [pdf(p, τ) for τ in τs]
        qs = [pdf(q, τ) for τ in τs]
        ws = ps ./ qs
        ws[Y .> γ] .= 0
        q = fit(typeof(q), τs, ws=ws)
    end
    return estimate(ImportanceSamplingEstimation(p, q, m), sys, ψ)
end
####################

#################### failure_probability 6
struct PopulationMonteCarloEstimation
    p         # nominal trajectory distribution
    qs        # vector of initial proposal distributions
    weighting # weighting scheme: ws = weighting(p, qs, τs)
    k_max     # number of iterations
end

function estimate(alg::PopulationMonteCarloEstimation, sys, ψ)
    p, qs, weighting = alg.p, alg.qs, alg.weighting
    k_max, m = alg.k_max, length(qs)    
    for k in 1:k_max
        τs = [rollout(sys, q) for q in qs]
        ws = [pdf(p, τ) * isfailure(ψ, τ) / pdf(q, τ) 
              for (q, τ) in zip(qs, τs)]
        resampler = Categorical(ws ./ sum(ws))
        qs = [proposal(qs[i], τs[i]) for i in rand(resampler, m)]
    end
    mis = MultipleImportanceSamplingEstimation(p, qs, weighting)
    return estimate(mis, sys, ψ)
end
####################

#################### failure_probability 7
struct SequentialMonteCarloEstimation
    p       # nominal trajectory distribution
    ḡs      # intermediate distributions
    perturb # τs′ = perturb(τs, ḡ)
    m       # number of samples
end

function estimate(alg::SequentialMonteCarloEstimation, sys, ψ)
    p, ḡs, perturb, m = alg.p, alg.ḡs, alg.perturb, alg.m
    p̄failure(τ) = isfailure(ψ, τ) * pdf(p, τ)
    τs = [rollout(sys, p) for i in 1:m]
    ws = [ḡs[1](τ) / p(τ) for τ in τs]
    for (ḡ, ḡ′) in zip(ḡs, [ḡs[2:end]...; p̄failure])
        τs′ = perturb(τs, ḡ)
        ws .*= [ḡ′(τ) / ḡ(τ) for τ in τs′]
        τs = τs′[rand(Categorical(ws ./ sum(ws)), m)]
        ws .= mean(ws)
    end
    return mean(ws)
end
####################

#################### failure_probability 8
function bridge_sampling_estimator(g₁τs, ḡ₁, g₂τs, ḡ₂, ḡb)
    ḡ₁s, ḡ₂s = ḡ₁.(g₁τs), ḡ₂.(g₂τs)
    ḡb₁s, ḡb₂s = ḡb.(g₁τs),  ḡb.(g₂τs)
    return mean(ḡb₂s ./ ḡ₂s) / mean(ḡb₁s ./ ḡ₁s)
end

function optimal_bridge(g₁τs, ḡ₁, g₂τs, ḡ₂, k_max)
    ratio = 1.0
    m₁, m₂ = length(g₁τs), length(g₂τs)
    ḡb(τ) = (ḡ₁(τ) * ḡ₂(τ)) / (m₁ * ḡ₁(τ) + m₂ * ratio * ḡ₂(τ))
    for k in k_max
        ratio = bridge_sampling_estimator(g₁τs, ḡ₁, g₂τs, ḡ₂, ḡb)
    end
    return ḡb
end
####################

#################### failure_probability 9
struct SelfImportanceSamplingEstimation
    p    # nominal distribution
    q̄    # unnormalized proposal density
    q̄_τs # samples from q̄
end

function estimate(alg::SelfImportanceSamplingEstimation, sys, ψ)
    p, q̄, q̄_τs = alg.p, alg.q̄, alg.q̄_τs
    ws = [pdf(p, τ) / q̄(τ) for τ in q̄_τs]
    ws ./= sum(ws)
    return mean(w * isfailure(ψ, τ) for (w, τ) in zip(ws, q̄_τs))
end
####################

#################### failure_probability 10
struct BridgeSamplingEstimation
    p       # nominal trajectory distribution
    ḡs      # intermediate distributions
    perturb # samples′ = perturb(samples, ḡ′)
    m       # number of samples from each intermediate distribution
    kb      # number of iterations for estimating optimal bridge
end

function estimate(alg::BridgeSamplingEstimation, sys, ψ)
    p, ḡs, perturb, m = alg.p, alg.ḡs, alg.perturb, alg.m
    p̄failure(τ) = isfailure(ψ, τ) * pdf(p, τ)
    τs = [rollout(sys, p) for i in 1:m]
    p̂fail = 1.0
    for (ḡ, ḡ′) in zip([p; ḡs...], [ḡs...; p̄failure])
        ws = [ḡ′(τ) / ḡ(τ) for τ in τs]
        τs′ = τs[rand(Categorical(ws ./ sum(ws)), m)]
        τs′ = perturb(τs′, ḡ′)
        ḡb = optimal_bridge(τs′, ḡ′, τs, ḡ, kb)
        ratio = bridge_sampling_estimator(τs′, ḡ′, τs, ḡ, ḡb)
        p̂fail *= ratio
        τs = τs′
    end
    return p̂fail
end
####################

#################### failure_probability 11
struct AdaptiveMultilevelSplitting
    p       # nominal trajectory distribution
    m       # number of samples
    m_elite # number of elite samples
    k_max   # maximum number of iterations
    f       # objective function f(τ, ψ) 
    perturb # τs′ = perturb(τs, p̄γ)
end

function estimate(alg::AdaptiveMultilevelSplitting, sys, ψ)
    p, m, m_elite, k_max = alg.p, alg.m, alg.m_elite, alg.k_max
    f, perturb = alg.f, alg.perturb
    τs = [rollout(sys, p) for i in 1:m]
    p̂fail = 1.0
    for i in 1:k_max
        Y = [f(τ, ψ) for τ in τs]
        order = sortperm(Y)
        γ = i == k_max ? 0 : max(0, Y[order[m_elite]])
        p̂fail *= mean(Y .≤ γ)
        γ == 0 && break
        τs = rand(τs[order[1:m_elite]], m)
        p̄γ(τ) = p(τ) * (f(τ, ψ) ≤ γ)
        τs = perturb(τs, p̄γ)
    end
    return p̂fail
end
####################

#################### forward_reachability 1
struct AvoidSetSpecification <: Specification
    set # avoid set
end
evaluate(ψ::AvoidSetSpecification, τ) = all(step.s ∉ ψ.set for step in τ)
####################

#################### forward_reachability 2
function get_matrices(sys)
    return Ts(sys.env), Ta(sys.env), Πo(sys.agent), Os(sys.sensor)
end

function linear_set_propagation(sys, 𝒮, 𝒳)
    Ts, Ta, Πo, Os = get_matrices(sys)
    return (Ts + Ta * Πo * Os) * 𝒮 ⊕ Ta * Πo * 𝒳.xo ⊕ Ta * 𝒳.xa ⊕ 𝒳.xs
end
####################

#################### forward_reachability 3
abstract type ReachabilityAlgorithm end

struct SetPropagation <: ReachabilityAlgorithm
    h # time horizon
end

function reachable(alg::SetPropagation, sys)
    h = alg.h
    𝒮, 𝒳 = 𝒮₁(sys.env), disturbance_set(sys)
    ℛ = 𝒮
    for t in 1:h
        𝒮 = linear_set_propagation(sys, 𝒮, 𝒳)
        ℛ = ℛ ∪ 𝒮
    end
    return ℛ
end
####################

#################### forward_reachability 4
¬(ψ::AvoidSetSpecification) = ψ.set
function satisfies(alg::SetPropagation, sys, ψ)
    ℛ = reachable(alg, sys)
    return !isempty(ℛ ∩ ¬ψ)
end
####################

#################### forward_reachability 5
struct OverapproximateSetPropagation <: ReachabilityAlgorithm
    h    # time horizon
    freq # overapproximation frequency
    ϵ    # overapproximation tolerance
end

function reachable(alg::OverapproximateSetPropagation, sys)
    h, freq, ϵ = alg.h, alg.freq, alg.ϵ
    𝒮, 𝒳 = 𝒮₁(sys.env), disturbance_set(sys)
    ℛ = 𝒮
    for t in 1:h
        𝒮 = linear_set_propagation(sys, 𝒮, 𝒳)
        ℛ = ℛ ∪ 𝒮
        𝒮 = t % freq == 0 ? overapproximate(𝒮, ϵ) : 𝒮
    end
    return ℛ
end
####################

#################### forward_reachability 6
Ab(𝒫) = tosimplehrep(constraints_list(𝒫))

function constrained_model(sys, d, 𝒮, 𝒳)
    model = Model(SCS.Optimizer)
    @variable(model, 𝐬[1:dim(𝒮),1:d])
    @variable(model, 𝐱o[1:dim(𝒳.xo),1:d])
    @variable(model, 𝐱s[1:dim(𝒳.xs),1:d])
    @variable(model, 𝐱a[1:dim(𝒳.xa),1:d])

    As, bs = Ab(𝒮)
    (Axo, bxo), (Axs, bxs), (Axa, bxa) = Ab(𝒳.xo), Ab(𝒳.xs), Ab(𝒳.xa)
    @constraint(model, As * 𝐬[:, 1] .≤ bs)
    for i in 1:d
        @constraint(model, Axo * 𝐱o[:, i] .≤ bxo)
        @constraint(model, Axs * 𝐱s[:, i] .≤ bxs)
        @constraint(model, Axa * 𝐱a[:, i] .≤ bxa)
    end

    Ts, Ta, Πo, Os = get_matrices(sys)
    for i in 1:d-1
        @constraint(model, (Ts + Ta*Πo*Os) * 𝐬[:, i] + Ta*Πo * 𝐱o[:, i] 
                            + Ta * 𝐱a[:, i] + 𝐱s[:, i] .== 𝐬[:, i+1])
    end
    return model
end

function ρ(model, 𝐝, d)
    𝐬 = model.obj_dict[:𝐬]
    @objective(model, Max, 𝐝' * 𝐬[:, d])
    optimize!(model)
    return objective_value(model)
end
####################

#################### forward_reachability 7
struct LinearProgramming <: ReachabilityAlgorithm
    h   # time horizon
    𝒟   # set of directions to evaluate support function
    tol # tolerance for checking satisfaction
end

function reachable(alg::LinearProgramming, sys)
    h, 𝒟 = alg.h, alg.𝒟
    𝒮, 𝒳 = 𝒮₁(sys.env), disturbance_set(sys)
    ℛ = 𝒮
    for d in 2:h
        model = constrained_model(sys, d, 𝒮, 𝒳)
        ρs = [ρ(model, 𝐝, d) for 𝐝 in 𝒟]
        ℛ = ℛ ∪ HPolytope([HalfSpace(𝐝, ρ) for (𝐝, ρ) in zip(𝒟, ρs)])
    end
    return ℛ
end
####################

#################### forward_reachability 8
function satisfies(alg::LinearProgramming, sys, ψ)
    𝒮, 𝒳 = 𝒮₁(sys.env), disturbance_set(sys)
    for d in 1:alg.h
        model = constrained_model(sys, d, 𝒮, 𝒳)
        @variable(model, u[1:dim(𝒮)])
        Au, bu = Ab(¬ψ)
        @constraint(model, Au * u .≤ bu)
        𝐬 = model.obj_dict[:𝐬]
        @objective(model, Min, sum((𝐬[i, d] - u[i])^2 for i in 1:dim(𝒮)))
        optimize!(model)
        if isapprox(objective_value(model), 0.0, atol=alg.tol)
            return false
        end
    end
    return true
end
####################

#################### nonlinear_reach 1
struct NaturalInclusion <: ReachabilityAlgorithm
    h # time horizon
end

function r(sys, x)
    s, 𝐱 = extract(sys.env, x)
    τ = rollout(sys, s, 𝐱)
    return τ[end].s
end

to_hyperrectangle(𝐈) = Hyperrectangle(low=[i.lo for i in 𝐈], 
                                      high=[i.hi for i in 𝐈])

function reachable(alg::NaturalInclusion, sys)
    𝐈′s = []
    for d in 1:alg.h
        𝐈 = intervals(sys, d)
        push!(𝐈′s, r(sys, 𝐈))
    end
    return UnionSetArray([to_hyperrectangle(𝐈′) for 𝐈′ in 𝐈′s])
end
####################

#################### nonlinear_reach 2
struct TaylorInclusion <: ReachabilityAlgorithm
    h     # time horizon
    order # order of Taylor inclusion function (supports 1 or 2)
end

function taylor_inclusion(sys, 𝐈, order)
    c = mid.(𝐈)
    fc = r(sys, c)
    if order == 1
        𝐈′ = [fc[i] + gradient(x->r(sys, x)[i], 𝐈)' * (𝐈 - c)
              for i in eachindex(fc)]
    else
        𝐈′ = [fc[i] + gradient(x->r(sys, x)[i], c)' * (𝐈 - c) + 
              (𝐈 - c)' * hessian(x->r(sys, x)[i], 𝐈) * (𝐈 - c)
              for i in eachindex(fc)]
    end
    return 𝐈′
end

function reachable(alg::TaylorInclusion, sys)
    𝐈′s = []
    for d in 1:alg.h
        𝐈 = intervals(sys, d)
        𝐈′ = taylor_inclusion(sys, 𝐈, alg.order)
        push!(𝐈′s, 𝐈′)
    end
    return UnionSetArray([to_hyperrectangle(𝐈′) for 𝐈′ in 𝐈′s])
end
####################

#################### nonlinear_reach 3
struct ConservativeLinearization <: ReachabilityAlgorithm
    h # time horizon
end

to_intervals(𝒫) = [interval(lo, hi) for (lo, hi) in zip(low(𝒫), high(𝒫))]

function conservative_linearization(sys, 𝒫)
    𝐈 = to_intervals(interval_hull(𝒫))
    c = mid.(𝐈)
    fc = r(sys, c)
    J = ForwardDiff.jacobian(x->r(sys, x), c)
    α = to_hyperrectangle([(𝐈 - c)'*hessian(x->r(sys, x)[i], 𝐈)*(𝐈 - c)
                           for i in eachindex(fc)])
    return fc + J * (𝒫 ⊕ -c) ⊕ α
end

function reachable(alg::ConservativeLinearization, sys)
    ℛs = []
    for d in 1:alg.h
        𝒮, 𝒳 = sets(sys, d)
        𝒮′ = conservative_linearization(sys, 𝒮 × 𝒳)
        push!(ℛs, 𝒮′)
    end
    return UnionSetArray([ℛs...])
end
####################

#################### nonlinear_reach 4
struct ConcreteTaylorInclusion <: ReachabilityAlgorithm
    h     # time horizon
    order # order of Taylor inclusion function (supports 1 or 2)
end

function reachable(alg::ConcreteTaylorInclusion, sys)
    𝐈 = intervals(sys, 2)
    s, _ = extract(sys.env, 𝐈)
    𝐈′s = [s]
    for d in 2:alg.h
        𝐈′ = taylor_inclusion(sys, 𝐈, alg.order)
        push!(𝐈′s, 𝐈′)
        s, _ = extract(sys.env, 𝐈)
        𝐈[1:length(s)] = s
    end
    return UnionSetArray([to_hyperrectangle(𝐈′) for 𝐈′ in 𝐈′s])
end
####################

#################### nonlinear_reach 5
struct ConcreteConservativeLinearization <: ReachabilityAlgorithm
    h # time horizon
end

function reachable(alg::ConcreteConservativeLinearization, sys)
    𝒮, 𝒳 = sets(sys, 2)
    ℛs = []
    push!(ℛs, 𝒮)
    for d in 2:alg.h
        𝒮 = conservative_linearization(sys, 𝒮 × 𝒳)
        push!(ℛs, 𝒮)
    end
    return UnionSetArray([ℛs...])
end
####################

#################### discrete_reachability 1
function to_graph(sys)
    𝒮 = states(sys.env)
    g = WeightedGraph(𝒮)
    for s in 𝒮
        𝒮′, ws = successors(sys, s)
        for (s′, w) in zip(𝒮′, ws)
            add_edge!(g, s, s′, w)
        end
    end
    return g
end
####################

#################### discrete_reachability 2
struct DiscreteForward <: ReachabilityAlgorithm
    h # time horizon
end

function reachable(alg::DiscreteForward, sys)
    g = to_graph(sys)
    𝒮 = 𝒮₁(sys.env)
    ℛ = 𝒮
    for d in 2:alg.h
        𝒮 = Set(reduce(vcat, [outneighbors(g, s) for s in 𝒮]))
        ℛ == (ℛ ∪ 𝒮) && break
        ℛ = ℛ ∪ 𝒮
    end
    return ℛ
end
####################

#################### discrete_reachability 3
struct DiscreteBackward <: ReachabilityAlgorithm
    h # time horizon
end

function backward_reachable(alg::DiscreteBackward, sys, ψ)
    g = to_graph(sys)
    𝒮 = ψ.set
    ℬ = 𝒮
    for d in 2:alg.h
        𝒮 = Set(reduce(vcat, [inneighbors(g, s) for s in 𝒮]))
        ℬ == (ℬ ∪ 𝒮) && break
        ℬ = ℬ ∪ 𝒮
    end
    return ℬ
end
####################

#################### discrete_reachability 4
struct ProbabilisticOccupancy <: ReachabilityAlgorithm
    h # time horizon
end

function reachable(alg::ProbabilisticOccupancy, sys)
    𝒮, g, dist = states(sys.env), to_graph(sys), Ps(sys.env)
    P = Dict(s => pdf(dist, s) for s in 𝒮)
    for t in 2:alg.h
        P = Dict(s => sum(get_weight(g, s′, s) * P[s′] 
                           for s′ in inneighbors(g, s)) for s in 𝒮)
    end
    return SetCategorical(P)
end
####################

#################### discrete_reachability 5
struct ProbabilisticFiniteHorizon <: ReachabilityAlgorithm
    h # time horizon
end

function reachable(alg::ProbabilisticFiniteHorizon, sys, ψ)
    𝒮, g, dist = states(sys.env), to_graph(sys), Ps(sys.env)
    𝒮T = ψ.set
    R = Dict(s => s ∈ 𝒮T ? 1.0 : 0.0 for s in 𝒮)
    for d in 2:alg.h
        R = Dict(s => s ∈ 𝒮T ? 1.0 : sum(get_weight(g, s, s′) * R[s′]
                 for s′ in outneighbors(g, s)) for s in 𝒮)
    end
    return sum(R[s] * pdf(dist, s) for s in 𝒮)
end
####################

#################### discrete_reachability 6
struct ProbabilisticInfiniteHorizon <: ReachabilityAlgorithm end

function reachable(alg::ProbabilisticInfiniteHorizon, sys, ψ)
    𝒮, g, dist = states(sys.env), to_graph(sys), Ps(sys.env)
    𝒮Ti = [index(g, s) for s in ψ.set]
    R₁ = [i ∈ 𝒮Ti ? 1.0 : 0.0 for i in eachindex(𝒮)]
    TR = to_matrix(g)
    TR[𝒮Ti, :] .= 0
    R∞ = (I - TR) \ R₁
    return sum(R∞[i] * pdf(dist, state(g, i)) for i in eachindex(𝒮))
end
####################

#################### explainability 1
struct Sensitivity
    x       # vector of trajectory inputs (s, 𝐱 = extract(sys.env, x))
    perturb # x′ = perturb(x, t)
    m       # number of samples per time step
end

function describe(alg::Sensitivity, sys, ψ)
    m, x, perturb = alg.m, alg.x, alg.perturb
    s, 𝐱 = extract(sys.env, x)
    τ = rollout(sys, s, 𝐱)
    ρ₀ = robustness([step.s for step in τ], ψ.formula)
    sensitivities = zeros(length(τ))
    for t in eachindex(τ)
        x′s = [perturb(x, t) for i in 1:m]
        τ′s = [rollout(sys, extract(sys.env, x′)...) for x′ in x′s]
        ρs = [robustness([st.s for st in τ′], ψ.formula) for τ′ in τ′s]
        sensitivities[t] = std(abs.(ρs .- ρ₀))
    end
    return sensitivities
end
####################

#################### explainability 2
struct GradientSensitivity
    x # vector of trajectory inputs (s, 𝐱 = extract(sys.env, x))
end

function describe(alg::GradientSensitivity, sys, ψ)
    function current_robustness(x)
        s, 𝐱 = extract(sys.env, x)
        τ = rollout(sys, s, 𝐱)
        return robustness([step.s for step in τ], ψ.formula)
    end

    return ForwardDiff.gradient(current_robustness, alg.x)
end
####################

#################### explainability 3
struct IntegratedGradients
    x # vector of trajectory inputs (s, 𝐱 = extract(sys.env, x))
    b # vector of baseline inputs
    m # number of steps for numerical integration
end

function describe(alg::IntegratedGradients, sys, ψ)
    function current_robustness(x)
        s, 𝐱 = extract(sys.env, x)
        τ = rollout(sys, s, 𝐱)
        return robustness([step.s for step in τ], ψ.formula)
    end
    αs = range(0, stop=1, length=alg.m)
    xs = [(1 - α) * alg.b .+ α * alg.x for α in αs]
    grads = [ForwardDiff.gradient(current_robustness, x) for x in xs]
    return mean(hcat(grads...), dims=2)
end
####################

#################### explainability 4
struct Shapley
    τ # current trajectory
    m # number of samples per time step
end

function shapley_rollout(sys, s, 𝐱, 𝐰, inds)
    τ = []
    for t in 1:length(𝐱)
        x = t ∈ inds ? 𝐱[t] : 𝐰[t]
        o, a, s′ = step(sys, s, x)
        push!(τ, (; s, o, a, x))
        s = s′
    end
    return τ
end

function describe(alg::Shapley, sys, ψ)
    τ, m = alg.τ, alg.m
    p =  NominalTrajectoryDistribution(sys, length(alg.τ))
    𝐱 = [step.x for step in τ]
    ϕs = zeros(length(τ))
    for t in eachindex(τ)
        for _ in 1:m
            𝐰 = [step.x for step in rollout(sys, p)]
            𝒫 = randperm(length(τ))
            j = findfirst(𝒫 .== t)
            τ₊ = shapley_rollout(sys, τ[1].s, 𝐱, 𝐰, 𝒫[1:j])
            τ₋ = shapley_rollout(sys, τ[1].s, 𝐱, 𝐰, 𝒫[1:j-1])
            ϕs[t] += robustness([step.s for step in τ₊], ψ.formula) - 
                     robustness([step.s for step in τ₋], ψ.formula)
        end
        ϕs[t] /= m
    end
    return ϕs
end
####################

#################### explainability 5
function counterfactual_objective(x, sys, ψ, x₀; ws=ones(3))
    s, 𝐱 = extract(sys.env, x)
    τ = rollout(sys, s, 𝐱)
    foutcome = robustness([step.s for step in τ], ψ.formula)
    fclose = -norm(x - x₀, 1)
    fplaus = logpdf(NominalTrajectoryDistribution(sys, length(𝐱)), τ)
    return ws' * [foutcome, fclose, fplaus]
end
####################

#################### explainability 6
struct Kmeans
    τs       # trajectories to cluster
    ϕ        # feature extraction function (x = ϕ(τ))
    d        # distance metric function (d(x[i], μⱼ))
    k        # number of clusters
    max_iter # maximum number of iterations
end

function describe(alg::Kmeans, sys, ψ)
    x = [alg.ϕ(τ) for τ in alg.τs]
    μ = x[randperm(length(x))[1:alg.k]]
    𝒞 = [Int[] for j in 1:alg.k]
    for _ in 1:alg.max_iter
        𝒞 = [Int[] for j in 1:alg.k]
        for i in eachindex(x)
            push!(𝒞[argmin([alg.d(x[i], μⱼ) for μⱼ in μ])], i)
        end
        for j in 1:alg.k
            if !isempty(𝒞[j])
                μ[j] = mean(x[i] for i in 𝒞[j])
            end
        end
    end
    return 𝒞, μ
end
####################

#################### runtime_monitoring 1
struct KNNMonitor
    data # ODD data matrix (each column is a datapoint)
    k    # number of neighbors
    γ    # threshold
end

function monitor(alg::KNNMonitor, input)
    kdtree = KDTree(alg.data)
    neighbors, distances = knn(kdtree, input, alg.k)
    return all(distances .< alg.γ)
end
####################

#################### runtime_monitoring 2
struct HullMonitor
    data # ODD data matrix (each column is a datapoint)
    𝒞    # collection of vectors containing cluster column indices
end

function monitor(alg::HullMonitor, input)
    for (k, v) in alg.𝒞
        hull = convex_hull([alg.data[:, i] for i in v])
        if input ∈ VPolytope(hull)
            return true
        end
    end
    return false
end
####################

#################### runtime_monitoring 3
struct SuperlevelSetMonitor
    dist # distribution
    γ    # likelihood threshold
end

function monitor(alg::SuperlevelSetMonitor, input)
    return pdf(alg.dist, input) > alg.γ
end
####################

#################### tuck_away 1
Base.rand(𝐝::Vector{DisturbanceDistribution}) = rand.(𝐝)

function Distributions.fit(d::DisturbanceDistribution, samples, w)
    𝐱_agent = [s.x.x_agent for s in samples]
    𝐱_env = [s.x.x_env for s in samples]
    𝐱_sensor = [s.x.x_sensor for s in samples]
    px_agent = fit(d.px_agent, 𝐱_agent, w)
    px_env = fit(d.px_env, 𝐱_env, w)
    px_sensor = fit(d.px_sensor, 𝐱_sensor, w)
    return DisturbanceDistribution(px_agent, px_env, px_sensor)
end

Distributions.fit(𝐝::Vector, samples, w) = [fit(d, [s[t] for s in samples], w) for (t, d) in enumerate(𝐝)]

Distributions.fit(d::Sampleable, samples, w::Missing) = fit(typeof(d), samples)
Distributions.fit(d::Sampleable, samples, w) = fit_mle(typeof(d), samples, w)

# function fit!(adv::Adversary, samples, w=missing)
#     adv.Ps = fit(adv.Ps, [s[1].s for s in samples], w)
#     adv.P𝐱 = fit(adv.P𝐱, samples, w)
#     return adv
# end

Distributions.pdf(𝐝::Vector, 𝐱::Vector) = prod(pdf(d, x) for (d, x) in zip(𝐝, 𝐱))
####################

#################### tuck_away 2
@with_kw struct WeightedGraph{T}
    g::SimpleWeightedDiGraph
    states2ind::Dict{T,Int} = Dict()
    ind2states::Dict{Int,T} = Dict()
end

function WeightedGraph{T}(nv::Int) where T
    return WeightedGraph{T}(SimpleWeightedDiGraph(nv), Dict(), Dict())
end

function WeightedGraph(vertices::Vector)
    nv = length(vertices)
    T = typeof(first(vertices))
    return WeightedGraph{T}(nv)
end

function SimpleWeightedGraphs.add_edge!(g::WeightedGraph{T}, si::T, sj::T, w::Real) where T
    val = values(g.states2ind)
    n = isempty(val) ? 0 : maximum(val)

    if !haskey(g.states2ind, si)
        n += 1
        g.states2ind[si] = n
    end
    i = g.states2ind[si]
    g.ind2states[i] = si

    if !haskey(g.states2ind, sj)
        n += 1
        g.states2ind[sj] = n
    end
    j = g.states2ind[sj]
    g.ind2states[j] = sj

    return add_edge!(g.g, i, j, w)    
end

function SimpleWeightedGraphs.outneighbors(g::WeightedGraph{T}, si::T) where T
    i = g.states2ind[si]
    J = outneighbors(g.g, i)
    return map(j->g.ind2states[j], J)
end

function SimpleWeightedGraphs.inneighbors(g::WeightedGraph{T}, si::T) where T
    i = g.states2ind[si]
    J = inneighbors(g.g, i)
    return map(j->g.ind2states[j], J)
end

function SimpleWeightedGraphs.edges(g::WeightedGraph{T}) where T
    E = []
    for edge in edges(g.g)
        (i,j,w) = Tuple(edge)
        si = g.ind2states[i]
        sj = g.ind2states[j]
        push!(E, (si, sj, w))
    end
    return E
end

function SimpleWeightedGraphs.get_weight(g::WeightedGraph{T}, si::T, sj::T) where T
    i = g.states2ind[si]
    j = g.states2ind[sj]
    return get_weight(g.g, i, j)
end

function to_matrix(g::WeightedGraph)
    𝒮 = keys(g.states2ind)
    n = length(𝒮)
    T = zeros(n, n)
    for s in 𝒮
        for s′ in outneighbors(g, s)
            T[g.states2ind[s], g.states2ind[s′]] = get_weight(g, s, s′)
        end
    end
    return T
end

index(g::WeightedGraph, s) = g.states2ind[s]
state(g::WeightedGraph, i) = g.ind2states[i]
####################

#################### problems 1
(env::Environment)(s, a, x) = env(s, a)
(sensor::Sensor)(s, x) = sensor(s)
(agent::Agent)(o, x) = agent(o)
####################

#################### problems 2
Ds(env::Environment, s, a) = Deterministic()
Da(agent::Agent, o) = Deterministic()
Do(sensor::Sensor, s) = Deterministic()
####################

#################### problems 3
struct SimpleGaussian <: Environment end
(env::SimpleGaussian)(s, a) = s
Ps(env::SimpleGaussian) = Normal()

struct NoAgent <: Agent end
(c::NoAgent)(s) = nothing

struct IdealSensor <: Sensor end
(sensor::IdealSensor)(s) = s
####################

#################### problems 4
struct MvGaussian <: Environment end
(env::MvGaussian)(s, a) = s
Ps(env::MvGaussian) = MvNormal(zeros(2), I)
####################

#################### problems 5
@with_kw struct MassSpringDamper <: Environment
    m = 1.0   # mass
    k = 10.0  # spring constant
    c = 2.0   # damping coefficient
    dt = 0.05 # time step
end

Ts(env::MassSpringDamper) = [1 env.dt; 
                             -env.k*env.dt/env.m 1-env.c*env.dt/env.m]
Ta(env::MassSpringDamper) = [0 env.dt/env.m]'
function (env::MassSpringDamper)(s, a)
    return Ts(env) * s + Ta(env) * a
end
Ps(env::MassSpringDamper) = Product([Uniform(-0.2, 0.2), 
                                     Uniform(-1e-12, 1e-12)])

struct AdditiveNoiseSensor <: Sensor
	Do # noise distribution
end

(sensor::AdditiveNoiseSensor)(s) = sensor(s, rand(Do(sensor, s)))
(sensor::AdditiveNoiseSensor)(s, x) = s + x
Do(sensor::AdditiveNoiseSensor, s) = sensor.Do
Os(sensor::AdditiveNoiseSensor) = I

struct ProportionalController <: Agent
    α # gain matrix (c = α' * o)
end
(c::ProportionalController)(o) = c.α' * o
Πo(agent::ProportionalController) = agent.α'
####################

#################### problems 6
@with_kw struct InvertedPendulum <: Environment
    m::Float64 = 1.0     # mass of the pendulum
    l::Float64 = 1.0     # length of the pendulum
    g::Float64 = 10.0    # acceleration due to gravity
    dt::Float64 = 0.05   # time step
    ω_max::Float64 = 8.0 # maximum angular velocity
    a_max::Float64 = 2.0 # maximum torque
end

function (env::InvertedPendulum)(s, a)
    θ, ω = s[1], s[2]
    dt, g, m, l = env.dt, env.g, env.m, env.l
    a = clamp(a, -env.a_max, env.a_max)
    ω = ω + (3g / (2 * l) * sin(θ) + 3 * a / (m * l^2)) * dt
    θ = θ + ω * dt
    ω = clamp(ω, -env.ω_max, env.ω_max)
    return [θ, ω]
end
Ps(env::InvertedPendulum) = Product([Uniform(-π / 16, π / 16), 
                                     Uniform(-1.0, 1.0)])
####################

#################### problems 7
@with_kw struct GridWorld <: Environment
    size = (10, 10)                          # dimensions of the grid
    terminal_states = [[5,5],[7,8]]          # goal and obstacle states
    directions = [[0,1],[0,-1],[-1,0],[1,0]] # up, down, left, right
    tprob = 0.7                              # probability do not slip
end

function Ds(env::GridWorld, s, a)
	slip_prob = (1 - env.tprob) / (length(env.directions) - 1)
	probs = fill(slip_prob, length(env.directions))
	probs[a] = env.tprob
	return Categorical(probs)
end
(env::GridWorld)(s, a) = env(s, a, rand(Ds(env, s, a)))
function (env::GridWorld)(s, a, x)
	if s in env.terminal_states
		return s
	else
		dir = env.directions[x]
		return clamp.(s .+ dir, [1, 1], env.size)
	end
end
Ps(env::GridWorld) = SetCategorical([[1, 1]])

struct DiscreteAgent <: Agent
	policy # dictionary mapping states to actions
end
(c::DiscreteAgent)(o) = c.policy[o]
####################

#################### problems 8
@with_kw struct ContinuumWorld <: Environment
   size = [10, 10]                          # dimensions
   terminal_centers = [[4.5,4.5],[6.5,7.5]] # obstacle and goal centers
   terminal_radii = [0.5, 0.5]              # radius of obstacle and goal
   directions = [[0,1],[0,-1],[-1,0],[1,0]] # up, down, left, right
   Σ = 0.5 * I(2)
end

Ds(env::ContinuumWorld, s, a) = MvNormal(zeros(2), env.Σ)
(env::ContinuumWorld)(s, a) = env(s, a, rand(Ds(env, s, a)))
function (env::ContinuumWorld)(s, a, x)
    is_terminal = [norm(s .- c) ≤ r 
			for (c, r) in zip(env.terminal_centers, env.terminal_radii)]
    if any(is_terminal)
		return s
	else
		dir = normalize(env.directions[a] .+ x)
		return clamp.(s .+ dir, [0, 0], env.size)
	end
end
Ps(env::ContinuumWorld) = SetCategorical([[0.5, 0.5]])

struct InterpAgent <: Agent
	grid # grid of discrete states using GridInteroplations.jl
	Q    # corresponding state-action values
end
(c::InterpAgent)(s) = argmax(interpolate(c.grid, q, s) for q in c.Q)
####################

#################### problems 9
@with_kw struct CollisionAvoidance <: Environment
	ddh_max::Float64 = 1.0                # maximum vertical acceleration
    𝒜::Vector{Float64} = [-5.0, 0.0, 5.0] # vertical rate commands
	Ds::Sampleable = Normal()             # vertical rate noise
end

Ds(env::CollisionAvoidance, s, a) = env.Ds
(env::CollisionAvoidance)(s, a) = env(s, a, rand(Ds(env, s, a)))
function (env::CollisionAvoidance)(s, a, x)
	a = env.𝒜[a]
	h, dh, a_prev, τ = s
	h = h + dh
    if a != 0.0
        if abs(a - dh) < env.ddh_max
            dh += a
        else
            dh += sign(a - dh) * env.ddh_max
        end
    end
    a_prev = a
    τ = max(τ - 1.0, -1.0)
	return [h, dh + x, a_prev, τ]
end
Ps(env::CollisionAvoidance) = product_distribution(Uniform(-100, 100), 
                                    Uniform(-10, 10), 
                                    DiscreteNonParametric([0], [1.0]), 
                                    DiscreteNonParametric([40], [1.0]))
####################

