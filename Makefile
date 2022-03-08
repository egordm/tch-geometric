SHELL=/bin/bash

ACTIVE_ENV=source ~/.bashrc
VENV_ACTIVATE=source $${VENV_ACTIVATE_SCRIPT}

develop:
	@$(ACTIVE_ENV); $(VENV_ACTIVATE); maturin develop

release:
	@$(ACTIVE_ENV); $(VENV_ACTIVATE); maturin develop --release