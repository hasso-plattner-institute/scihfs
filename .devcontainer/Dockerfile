FROM python:3.12.2-slim

# Update package list and install necessary packages
# hadolint ignore DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends git zsh curl wget ssh-client manpages-dev libgmp-dev libcdd-dev python3-sphinx pandoc build-essential graphviz graphviz-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure shell to accept piping
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Lock poetry version and install directory; install poetry and add it to $PATH (for any ad-hoc use inside container)
ENV POETRY_VERSION=1.8.4
ENV POETRY_HOME=/etc/poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=${POETRY_HOME} POETRY_VERSION=${POETRY_VERSION} python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Create and switch to /hfs as working directory
RUN mkdir /hfs
WORKDIR /hfs

# Make available the dependency files inside the container
COPY poetry.lock pyproject.toml ./

# Install all required packages with poetry, including the "dev" group packages
RUN poetry config virtualenvs.create false \
    && poetry lock --no-update \
    && poetry install --no-interaction --no-root --no-cache --quiet --sync

# Install extensions for zsh for nicer visuals when using the container terminal
# hadolint ignore DL2086
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" \
    && git clone https://github.com/romkatv/powerlevel10k.git ~/.oh-my-zsh/custom/themes/powerlevel10k \
    && git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting \
    && git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# Initialize zsh configuration
COPY .devcontainer/.zshrc /root
COPY .devcontainer/.p10k.zsh /root
RUN /bin/zsh /root/.zshrc

# For the case of running a Jupyter notebook service inside the container, this is a requirement to make it available outside of it
EXPOSE 8888

# Entrypoint for the container
CMD ["/bin/zsh"]
