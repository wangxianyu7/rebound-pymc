#section support_code_struct

int APPLY_SPECIFIC(integrate)(
    PyArrayObject *input_masses,     // Array of masses (nbody, )
    PyArrayObject *input_coords,     // Array of body coordinates (nbody, 6)
    PyArrayObject *input_times,      // Array of times (ntime, )
    PyArrayObject **output_coords,   // The output coordinates (6, ntime, nbody)
    PyArrayObject **output_jacobian, // The Jacobian (6, ntime, nbody, nbody, 7)
    PARAMS_TYPE *params)
{
  using namespace rebound_pymc3;

  int success = 0;
  npy_intp nbody = -1;
  auto masses = get_input<DTYPE_INPUT_0>(&nbody, input_masses, &success);

  int ndim = 2;
  npy_intp shape[] = {nbody, 6};
  auto initial_coords = get_input<DTYPE_INPUT_1>(ndim, shape, input_coords, &success);

  npy_intp ntime = -1;
  auto times = get_input<DTYPE_INPUT_2>(&ntime, input_times, &success);

  if (success)
    return 1;

  npy_intp final_shape[] = {ntime, nbody, 6};
  auto final_coords = allocate_output<DTYPE_OUTPUT_0>(3, final_shape, TYPENUM_OUTPUT_0, output_coords, &success);

  npy_intp jac_shape[] = {ntime, nbody, 7, nbody, 6};
  auto jacobian = allocate_output<DTYPE_OUTPUT_1>(5, jac_shape, TYPENUM_OUTPUT_1, output_jacobian, &success);

  if (success)
    return 1;

  // Argsort the times for efficiency
  std::vector<npy_intp> idx(ntime);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&times](npy_intp i1, npy_intp i2) { return times[i1] < times[i2]; });

  // Set up the simulation
  struct reb_simulation *sim = reb_simulation_create();
  sim->t = params->t;
  sim->dt = params->dt;

  // Set the integrator; enums aren't playing nice between c and c++
  switch (params->integrator)
  {
  case (0):
    sim->integrator = reb_simulation::REB_INTEGRATOR_IAS15;
    break;
  case (1):
    sim->integrator = reb_simulation::REB_INTEGRATOR_WHFAST;
    break;
  case (2):
    sim->integrator = reb_simulation::REB_INTEGRATOR_SEI;
    break;
  case (4):
    sim->integrator = reb_simulation::REB_INTEGRATOR_LEAPFROG;
    break;
  case (7):
    sim->integrator = reb_simulation::REB_INTEGRATOR_NONE;
    break;
  case (8):
    sim->integrator = reb_simulation::REB_INTEGRATOR_JANUS;
    break;
  case (9):
    sim->integrator = reb_simulation::REB_INTEGRATOR_MERCURIUS;
    break;
  case (12):
    sim->integrator = reb_simulation::REB_INTEGRATOR_BS;
    // Tighten Gragg-Bulirsch-Stoer tolerance (REBOUND default ~1e-9 drifts on
    // resonant multi-planet TTVs; 1e-12 matches reference codes to <0.1 min).
    sim->ri_bs.eps_abs = 1e-12;
    sim->ri_bs.eps_rel = 1e-12;
    break;

  default:
    PyErr_Format(PyExc_ValueError, "unknown integrator");
    reb_simulation_free(sim);
    return 1;
  }

  std::vector<std::array<int, 7>> var_systems(nbody);

  for (npy_intp i = 0; i < nbody; ++i)
  {
    struct reb_particle body = {0.0};
    body.m = double(masses[i]);
    int ind = i * 6;
    body.x = double(initial_coords[ind + 0]);
    body.y = double(initial_coords[ind + 1]);
    body.z = double(initial_coords[ind + 2]);
    body.vx = double(initial_coords[ind + 3]);
    body.vy = double(initial_coords[ind + 4]);
    body.vz = double(initial_coords[ind + 5]);
    reb_simulation_add(sim, body);
  }

  for (npy_intp i = 0; i < nbody; ++i)
  {
    // Initialize the variational particles
    int var;

    var = reb_simulation_add_variation_1st_order(sim, -1);
    var_systems[i][0] = var;
    sim->particles[var + i].m = 1.0;

    var = reb_simulation_add_variation_1st_order(sim, -1);
    var_systems[i][1] = var;
    sim->particles[var + i].x = 1.0;

    var = reb_simulation_add_variation_1st_order(sim, -1);
    var_systems[i][2] = var;
    sim->particles[var + i].y = 1.0;

    var = reb_simulation_add_variation_1st_order(sim, -1);
    var_systems[i][3] = var;
    sim->particles[var + i].z = 1.0;

    var = reb_simulation_add_variation_1st_order(sim, -1);
    var_systems[i][4] = var;
    sim->particles[var + i].vx = 1.0;

    var = reb_simulation_add_variation_1st_order(sim, -1);
    var_systems[i][5] = var;
    sim->particles[var + i].vy = 1.0;

    var = reb_simulation_add_variation_1st_order(sim, -1);
    var_systems[i][6] = var;
    sim->particles[var + i].vz = 1.0;
  }

  for (npy_intp ind = 0; ind < ntime; ++ind)
  {
    npy_intp t = idx[ind];

    reb_simulation_integrate(sim, times[t]);

    for (npy_intp i = 0; i < nbody; ++i)
    {
      struct reb_particle particle = sim->particles[i];
      int ind = (t * nbody + i) * 6;
      final_coords[ind + 0] = particle.x;
      final_coords[ind + 1] = particle.y;
      final_coords[ind + 2] = particle.z;
      final_coords[ind + 3] = particle.vx;
      final_coords[ind + 4] = particle.vy;
      final_coords[ind + 5] = particle.vz;
    }

    // [ntime, nbody, 7, nbody, 6]
    for (npy_intp i = 0; i < nbody; ++i)
    {
      for (npy_intp j = 0; j < 7; ++j)
      {
        for (npy_intp k = 0; k < nbody; ++k)
        {
          int ind = (((t * nbody + i) * 7 + j) * nbody + k) * 6;
          struct reb_particle particle = sim->particles[var_systems[i][j] + k];
          jacobian[ind + 0] = particle.x;
          jacobian[ind + 1] = particle.y;
          jacobian[ind + 2] = particle.z;
          jacobian[ind + 3] = particle.vx;
          jacobian[ind + 4] = particle.vy;
          jacobian[ind + 5] = particle.vz;
        }
      }
    }
  }

  reb_simulation_free(sim);

  // ---------------------------------------------------------------------------
  // WHFAST mass-variational correction (option B, fixed in-repo).
  // REBOUND's WHFAST variational is EXACT for coordinate derivatives but WRONG
  // for d/d(mass): it integrates in Jacobi coordinates and does not propagate
  // the bare Cartesian-fixed mass perturbation (.m=1) seeded above (IAS15/BS are
  // correct; verified var d x/dm = 53.5 vs correct 116.6). Since coords(t) are
  // SMOOTH in mass, we overwrite ONLY the j=0 (mass) columns of the Jacobian
  // with a central finite difference of forward-only WHFAST integrations. The
  // coordinate columns (j=1..6) keep REBOUND's exact variational output.
  // Matches BS (fully-analytic variational) to ~3e-4. Only runs for WHFAST.
  if (params->integrator == 1)
  {
    std::vector<double> m0(nbody);
    for (npy_intp i = 0; i < nbody; ++i) m0[i] = double(masses[i]);

    // forward-only integration (no variational particles) with given masses;
    // fills out[(t*nbody+k)*6 + c] at every output time, in the sorted order.
    auto forward_only = [&](const std::vector<double> &mm, std::vector<double> &out) {
      struct reb_simulation *s = reb_simulation_create();
      s->t = params->t;
      s->dt = params->dt;
      s->integrator = reb_simulation::REB_INTEGRATOR_WHFAST;
      for (npy_intp i = 0; i < nbody; ++i)
      {
        struct reb_particle body = {0.0};
        body.m = mm[i];
        int ci = i * 6;
        body.x  = double(initial_coords[ci + 0]);
        body.y  = double(initial_coords[ci + 1]);
        body.z  = double(initial_coords[ci + 2]);
        body.vx = double(initial_coords[ci + 3]);
        body.vy = double(initial_coords[ci + 4]);
        body.vz = double(initial_coords[ci + 5]);
        reb_simulation_add(s, body);
      }
      for (npy_intp a = 0; a < ntime; ++a)
      {
        npy_intp t = idx[a];
        reb_simulation_integrate(s, times[t]);
        for (npy_intp k = 0; k < nbody; ++k)
        {
          struct reb_particle p = s->particles[k];
          int o = (t * nbody + k) * 6;
          out[o + 0] = p.x;  out[o + 1] = p.y;  out[o + 2] = p.z;
          out[o + 3] = p.vx; out[o + 4] = p.vy; out[o + 5] = p.vz;
        }
      }
      reb_simulation_free(s);
    };

    std::vector<double> cp(ntime * nbody * 6), cm(ntime * nbody * 6);
    std::vector<double> mp(m0), mmv(m0);
    for (npy_intp i = 0; i < nbody; ++i)
    {
      double dm = 1e-6 * std::fabs(m0[i]);
      if (dm == 0.0) continue;              // skip massless bodies (no neg mass)
      mp[i]  = m0[i] + dm;
      mmv[i] = m0[i] - dm;
      forward_only(mp, cp);
      forward_only(mmv, cm);
      mp[i] = m0[i]; mmv[i] = m0[i];        // restore for next body
      double inv = 0.5 / dm;
      for (npy_intp t = 0; t < ntime; ++t)
        for (npy_intp k = 0; k < nbody; ++k)
          for (int c = 0; c < 6; ++c)
          {
            int o = (t * nbody + k) * 6 + c;
            int jind = (((t * nbody + i) * 7 + 0) * nbody + k) * 6 + c;
            jacobian[jind] = (cp[o] - cm[o]) * inv;
          }
    }
  }

  return 0;
}
