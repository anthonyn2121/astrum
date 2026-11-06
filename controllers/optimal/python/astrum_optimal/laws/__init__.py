"""Control laws for the ASTRUM optimal-control controller, each an
:class:`astrum_client.runtime.Law`: ``lqr`` (:class:`~astrum_optimal.laws.lqr.LqrLaw`),
``mpc`` (:class:`~astrum_optimal.laws.mpc.MpcLaw`), ``figure8``
(:class:`~astrum_optimal.laws.figure8.Figure8Law`) and ``formation``
(:class:`~astrum_optimal.laws.formation.FormationLaw`). ``run.py`` holds the law
table; modules whose name starts with ``_`` are shared helpers, not laws.
"""
