using SparseArrays, PyPlot, Printf, JLD2, KrylovKit, LinearMaps, LinearAlgebra, PyCall, FFTW, Random

@pyimport cmasher

rc("axes", titlesize="medium")

dipole(x, y) = 1/2*(sin(x) - sin(y))
onedim(x, y) = sin(x)
function gaussian(x, y)
  g = 0
  for i = -3:3
    for j = -3:3
      g += 1/2*exp(-(x-4i*π)^2/2-(y-4j*π)^2/2)
    end
  end
  return g
end

function random(n)
  # make random streamfunction, ensuring that all shared Fourier coefficients are the same at different resolutions
  Random.seed!(3)
  a = zeros(Float64, n÷2+1, n)
  θ = zeros(ComplexF64, n÷2+1, n)
  for i = 0:n÷2
    for j = 0:i-1
      a[i+1,j+1] = randn()/(1+i^2+j^2)^(5/2)
      θ[i+1,j+1] = rand()
    end
    for j = -i:-1
      a[i+1,n+j+1] = randn()/(1+i^2+j^2)^(5/2)
      θ[i+1,n+j+1] = rand()
    end
    if i < n÷2
      for k = 0:i
        a[k+1,i+1] = randn()/(1+k^2+i^2)^(5/2)
        θ[k+1,i+1] = rand()
      end
    end
    for k = 0:i
      a[k+1,n-i] = randn()/(1+k^2+i^2)^(5/2)
      θ[k+1,n-i] = rand()
    end
  end
  ψ̂ = a.*exp.(2π*im*θ)*n^2
  ψ = irfft(ψ̂, n)
  return ψ
end

function ybjmodes(flow, n, ε, tag; adv=true, nev=100, tol=1e-12, order=2, domain=[0, 2π])

  # ∂_t ϕ + J(ψ, ϕ) - i/2 ε^2 Δϕ + iζ/2 ϕ = 0
  # i ∂_t ϕ = -ε^2/2 Δϕ + ζ/2 ϕ - i J(ψ, ϕ)

  Δ = (domain[2]-domain[1])/n
  x = domain[1] .+ (1:n)*Δ
  y = domain[1] .+ (1:n)*Δ

  if flow == dipole || flow == onedim || flow == gaussian
    ψ = reshape(flow.(x, y'), n^2)
  else
    ψ = reshape(flow(n), n^2)
  end

  fl = @sprintf("/groups/oceanphysics/ybjmodes/eig_%s_%d_%7.2e_%04d_%d_%04d.jld2", tag, adv, ε, n, order, nev)
  println(fl)

  λ = nothing
  ϕ = nothing

  try

    λ = load(fl, "λ")
    ϕ = load(fl, "ϕ")

  catch e

    idx(i, j) = mod1(i, n) + (mod1(j, n)-1)*n

    I = Int[]; J = Int[]; V = adv ? ComplexF64[] : Float64[]
    for i = 1:n, j = 1:n
      if order == 2
        # dispersion
        push!(I, idx(i, j)); push!(J, idx(i, j)); push!(V, 2ε^2/Δ^2)
        push!(I, idx(i, j)); push!(J, idx(i-1, j)); push!(V, -ε^2/2Δ^2)
        push!(I, idx(i, j)); push!(J, idx(i+1, j)); push!(V, -ε^2/2Δ^2)
        push!(I, idx(i, j)); push!(J, idx(i, j-1)); push!(V, -ε^2/2Δ^2)
        push!(I, idx(i, j)); push!(J, idx(i, j+1)); push!(V, -ε^2/2Δ^2)
        # refraction
        ψxx = (ψ[idx(i-1, j)] - 2ψ[idx(i, j)] + ψ[idx(i+1, j)])/Δ^2
        ψyy = (ψ[idx(i, j-1)] - 2ψ[idx(i, j)] + ψ[idx(i, j+1)])/Δ^2
        push!(I, idx(i, j)); push!(J, idx(i, j)); push!(V, (ψxx+ψyy)/2)
        # 2nd order Arakawa for advection (J₁)
        if adv
          # J++/3
          push!(I, idx(i, j)); push!(J, idx(i+1, j)); push!(V, +1im*(ψ[idx(i, j+1)] - ψ[idx(i, j-1)])/4Δ^2/3)
          push!(I, idx(i, j)); push!(J, idx(i-1, j)); push!(V, -1im*(ψ[idx(i, j+1)] - ψ[idx(i, j-1)])/4Δ^2/3)
          push!(I, idx(i, j)); push!(J, idx(i, j+1)); push!(V, -1im*(ψ[idx(i+1, j)] - ψ[idx(i-1, j)])/4Δ^2/3)
          push!(I, idx(i, j)); push!(J, idx(i, j-1)); push!(V, +1im*(ψ[idx(i+1, j)] - ψ[idx(i-1, j)])/4Δ^2/3)
          # J+×/3
          push!(I, idx(i, j)); push!(J, idx(i+1, j)); push!(V, +1im*(ψ[idx(i+1, j+1)] - ψ[idx(i+1, j-1)])/4Δ^2/3)
          push!(I, idx(i, j)); push!(J, idx(i-1, j)); push!(V, -1im*(ψ[idx(i-1, j+1)] - ψ[idx(i-1, j-1)])/4Δ^2/3)
          push!(I, idx(i, j)); push!(J, idx(i, j+1)); push!(V, -1im*(ψ[idx(i+1, j+1)] - ψ[idx(i-1, j+1)])/4Δ^2/3)
          push!(I, idx(i, j)); push!(J, idx(i, j-1)); push!(V, +1im*(ψ[idx(i+1, j-1)] - ψ[idx(i-1, j-1)])/4Δ^2/3)
          # J×+/3
          push!(I, idx(i, j)); push!(J, idx(i+1, j+1)); push!(V, +1im*(ψ[idx(i, j+1)] - ψ[idx(i+1, j)])/4Δ^2/3)
          push!(I, idx(i, j)); push!(J, idx(i-1, j-1)); push!(V, -1im*(ψ[idx(i-1, j)] - ψ[idx(i, j-1)])/4Δ^2/3)
          push!(I, idx(i, j)); push!(J, idx(i-1, j+1)); push!(V, -1im*(ψ[idx(i, j+1)] - ψ[idx(i-1, j)])/4Δ^2/3)
          push!(I, idx(i, j)); push!(J, idx(i+1, j-1)); push!(V, +1im*(ψ[idx(i+1, j)] - ψ[idx(i, j-1)])/4Δ^2/3)
        end
      elseif order == 4
        # dispersion
        push!(I, idx(i, j)); push!(J, idx(i, j)); push!(V, 5ε^2/2Δ^2)
        push!(I, idx(i, j)); push!(J, idx(i-2, j)); push!(V, ε^2/24Δ^2)
        push!(I, idx(i, j)); push!(J, idx(i-1, j)); push!(V, -2ε^2/3Δ^2)
        push!(I, idx(i, j)); push!(J, idx(i+1, j)); push!(V, -2ε^2/3Δ^2)
        push!(I, idx(i, j)); push!(J, idx(i+2, j)); push!(V, ε^2/24Δ^2)
        push!(I, idx(i, j)); push!(J, idx(i, j-2)); push!(V, ε^2/24Δ^2)
        push!(I, idx(i, j)); push!(J, idx(i, j-1)); push!(V, -2ε^2/3Δ^2)
        push!(I, idx(i, j)); push!(J, idx(i, j+1)); push!(V, -2ε^2/3Δ^2)
        push!(I, idx(i, j)); push!(J, idx(i, j+2)); push!(V, ε^2/24Δ^2)
        # refraction
        ψxx = (-ψ[idx(i-2, j)]/12 + 4ψ[idx(i-1, j)]/3 - 5ψ[idx(i, j)]/2 + 4ψ[idx(i+1, j)]/3 - ψ[idx(i+2, j)]/12)/Δ^2
        ψyy = (-ψ[idx(i, j-2)]/12 + 4ψ[idx(i, j-1)]/3 - 5ψ[idx(i, j)]/2 + 4ψ[idx(i, j+1)]/3 - ψ[idx(i, j+2)]/12)/Δ^2
        push!(I, idx(i, j)); push!(J, idx(i, j)); push!(V, (ψxx + ψyy)/2)
        # 4nd order Arakawa for advection (2J₁ - J₂)
        if adv
          # J₁
          # 2J++/3
          push!(I, idx(i, j)); push!(J, idx(i+1, j)); push!(V, +1im*(ψ[idx(i, j+1)] - ψ[idx(i, j-1)])/4Δ^2*2/3)
          push!(I, idx(i, j)); push!(J, idx(i-1, j)); push!(V, -1im*(ψ[idx(i, j+1)] - ψ[idx(i, j-1)])/4Δ^2*2/3)
          push!(I, idx(i, j)); push!(J, idx(i, j+1)); push!(V, -1im*(ψ[idx(i+1, j)] - ψ[idx(i-1, j)])/4Δ^2*2/3)
          push!(I, idx(i, j)); push!(J, idx(i, j-1)); push!(V, +1im*(ψ[idx(i+1, j)] - ψ[idx(i-1, j)])/4Δ^2*2/3)
          # 2J+×/3
          push!(I, idx(i, j)); push!(J, idx(i+1, j)); push!(V, +1im*(ψ[idx(i+1, j+1)] - ψ[idx(i+1, j-1)])/4Δ^2*2/3)
          push!(I, idx(i, j)); push!(J, idx(i-1, j)); push!(V, -1im*(ψ[idx(i-1, j+1)] - ψ[idx(i-1, j-1)])/4Δ^2*2/3)
          push!(I, idx(i, j)); push!(J, idx(i, j+1)); push!(V, -1im*(ψ[idx(i+1, j+1)] - ψ[idx(i-1, j+1)])/4Δ^2*2/3)
          push!(I, idx(i, j)); push!(J, idx(i, j-1)); push!(V, +1im*(ψ[idx(i+1, j-1)] - ψ[idx(i-1, j-1)])/4Δ^2*2/3)
          # 2J×+/3
          push!(I, idx(i, j)); push!(J, idx(i+1, j+1)); push!(V, +1im*(ψ[idx(i, j+1)] - ψ[idx(i+1, j)])/4Δ^2*2/3)
          push!(I, idx(i, j)); push!(J, idx(i-1, j-1)); push!(V, -1im*(ψ[idx(i-1, j)] - ψ[idx(i, j-1)])/4Δ^2*2/3)
          push!(I, idx(i, j)); push!(J, idx(i-1, j+1)); push!(V, -1im*(ψ[idx(i, j+1)] - ψ[idx(i-1, j)])/4Δ^2*2/3)
          push!(I, idx(i, j)); push!(J, idx(i+1, j-1)); push!(V, +1im*(ψ[idx(i+1, j)] - ψ[idx(i, j-1)])/4Δ^2*2/3)
          # J₂
          # -J××/3
          push!(I, idx(i, j)); push!(J, idx(i+1, j+1)); push!(V, -1im*(ψ[idx(i-1, j+1)] - ψ[idx(i+1, j-1)])/8Δ^2/3)
          push!(I, idx(i, j)); push!(J, idx(i-1, j-1)); push!(V, +1im*(ψ[idx(i-1, j+1)] - ψ[idx(i+1, j-1)])/8Δ^2/3)
          push!(I, idx(i, j)); push!(J, idx(i-1, j+1)); push!(V, +1im*(ψ[idx(i+1, j+1)] - ψ[idx(i-1, j-1)])/8Δ^2/3)
          push!(I, idx(i, j)); push!(J, idx(i+1, j-1)); push!(V, -1im*(ψ[idx(i+1, j+1)] - ψ[idx(i-1, j-1)])/8Δ^2/3)
          # -J×+/3
          push!(I, idx(i, j)); push!(J, idx(i+1, j+1)); push!(V, -1im*(ψ[idx(i, j+2)] - ψ[idx(i+2, j)])/8Δ^2/3)
          push!(I, idx(i, j)); push!(J, idx(i-1, j-1)); push!(V, +1im*(ψ[idx(i-2, j)] - ψ[idx(i, j-2)])/8Δ^2/3)
          push!(I, idx(i, j)); push!(J, idx(i-1, j+1)); push!(V, +1im*(ψ[idx(i, j+2)] - ψ[idx(i-2, j)])/8Δ^2/3)
          push!(I, idx(i, j)); push!(J, idx(i+1, j-1)); push!(V, -1im*(ψ[idx(i+2, j)] - ψ[idx(i, j-2)])/8Δ^2/3)
          # -J+×/3
          push!(I, idx(i, j)); push!(J, idx(i+2, j)); push!(V, -1im*(ψ[idx(i+1, j+1)] - ψ[idx(i+1, j-1)])/8Δ^2/3)
          push!(I, idx(i, j)); push!(J, idx(i-2, j)); push!(V, +1im*(ψ[idx(i-1, j+1)] - ψ[idx(i-1, j-1)])/8Δ^2/3)
          push!(I, idx(i, j)); push!(J, idx(i, j+2)); push!(V, +1im*(ψ[idx(i+1, j+1)] - ψ[idx(i-1, j+1)])/8Δ^2/3)
          push!(I, idx(i, j)); push!(J, idx(i, j-2)); push!(V, -1im*(ψ[idx(i+1, j-1)] - ψ[idx(i-1, j-1)])/8Δ^2/3)
        end
      else
        throw(ArgumentError(order, "Order must be 2 or 4."))
      end
    end
    H = sparse(I, J, V)

    if adv
      λ, ϕ = eigsolve(InverseMap(lu(H)), n^2, nev, :LM; krylovdim=2nev+1, verbosity=3, ishermitian=true, tol)
    else                                                                    
      λ, ϕ = eigsolve(InverseMap(lu(H)), n^2, nev, :LM; krylovdim=2nev+1, verbosity=3, issymmetric=true, tol)
    end
    λ = λ.^-1
    ϕ = cat(ϕ...; dims=2)
    ϕ ./= Δ*sign.(real.(ones(n^2)'*ϕ))

    save(fl, "ϕ", ϕ, "λ", λ)

  end

  a = ϕ'*ones(n^2)*Δ^2

  λ = real.(λ)

  N = 10

  gtm = sortperm(abs.(a).^2; rev=true)
  srt = gtm[sortperm(λ[gtm[1:N]])]

  cl = 1.115

  function πlabel(x)
    if x == 0
      return "0"
    elseif x == 1
      return "π"
    elseif x == -1
      return "\$-π\$"
    else
      return @sprintf("\$%dπ\$", x)
    end
  end

  ioff()
  fig, ax = subplots(N, 2; sharex=true, sharey=true, figsize=(4.8, N*2.4))
  for i = 1:size(ax, 1)
    ax[i,1].imshow(reshape(real.(ϕ[:,srt[i]]), (n, n))'; origin="lower", vmin=-cl, vmax=cl, cmap=cmasher.fusion.reversed(), extent=[domain[1], domain[2], domain[1], domain[2]])
    ax[i,2].imshow(reshape(imag.(ϕ[:,srt[i]]), (n, n))'; origin="lower", vmin=-cl, vmax=cl, cmap=cmasher.fusion.reversed(), extent=[domain[1], domain[2], domain[1], domain[2]])
    ax[i,1].contour(x, y, reshape(ψ .- sum(ψ)/n^2, (n, n))'; levels=-2:0.2:2, colors="gray", linewidths=0.8)
    ax[i,2].contour(x, y, reshape(ψ .- sum(ψ)/n^2, (n, n))'; levels=-2:0.2:2, colors="gray", linewidths=0.8)
    ax[i,1].set_title(@sprintf("\$|a_{%d}|^2 = {%5.3f}\$", i, abs(a[srt[i]])^2/(domain[2]-domain[1])^2); loc="center")
    ax[i,2].set_title(@sprintf("\$λ_{%d} = {%5.3f}\$", i, λ[srt[i]]); loc="center")
  end
  ax[end,1].set_xticks((round(domain[1]/π):round(domain[2]/π))*π)
  ax[end,1].set_yticks((round(domain[1]/π):round(domain[2]/π))*π)
  ax[end,1].set_xticklabels(πlabel.(round(domain[1]/π):round(domain[2]/π)))
  ax[end,1].set_yticklabels(πlabel.(round(domain[1]/π):round(domain[2]/π)))
  fig.tight_layout()
  fig.savefig(@sprintf("modes_%s_%d_%7.2e_%04d_%d_%04d.pdf", tag, adv, ε, n, order, nev), dpi=800)
  ion()

  return λ, ϕ, a

end
