FROM python:3.13.12

# Update package list and install necessary packages
# hadolint ignore DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends zsh ssh-client manpages-dev libcdd-dev python3-sphinx pandoc build-essential graphviz graphviz-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure shell to accept piping
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set locale so Sphinx (and other tools) don't fail on locale.setlocale()
ENV LANG=C.UTF-8

# Install uv and use Python from base image
COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /uvx /bin/
ENV UV_SYSTEM_PYTHON=1

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

# Create and switch to /scihfs as working directory
WORKDIR /scihfs

# Make available the dependency files inside the container
COPY uv.lock pyproject.toml ./

# Install all required packages with uv, including the "dev" group packages
RUN uv sync --no-install-project --frozen

# Entrypoint for the container
CMD ["/bin/zsh"]
