FROM python:3.13.12-slim

LABEL org.opencontainers.image.source="https://github.com/hasso-plattner-institute/scihfs"

# Configure shell to accept piping
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Update package list and install necessary packages
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git zsh ssh-client libcdd-dev python3-sphinx pandoc build-essential sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set locale so Sphinx (and other tools) don't fail on locale.setlocale()
ENV LANG=C.UTF-8

# Install uv and use Python from uv base image
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /bin/
ENV UV_SYSTEM_PYTHON=1

# Install extensions for zsh for nicer visuals when using the container terminal
# hadolint ignore=DL2086
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" \
    && git clone --depth 1 https://github.com/romkatv/powerlevel10k.git ~/.oh-my-zsh/custom/themes/powerlevel10k \
    && git clone --depth 1 https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting \
    && git clone --depth 1 https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
    && rm -rf ~/.oh-my-zsh/.git \
        ~/.oh-my-zsh/custom/themes/powerlevel10k/.git \
        ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting/.git \
        ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions/.git

# Initialize zsh configuration
COPY .devcontainer/.zshrc .devcontainer/.p10k.zsh /root/
RUN /bin/zsh /root/.zshrc

# Create and switch to /scihfs as working directory
WORKDIR /scihfs

# Make the dependency files available inside the container
COPY uv.lock pyproject.toml ./

# Install all required packages with uv, including the "dev" group packages
RUN uv sync --no-install-project --frozen

# Create non-root user with passwordless sudo, add zsh to its home directory
ARG USERNAME=nonroot
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && cp -r /root/.oh-my-zsh /home/nonroot/.oh-my-zsh \
    && cp /root/.zshrc /root/.p10k.zsh /home/nonroot/ \
    && chown -R $USER_UID:$USER_GID /home/nonroot

# Switch to non-root user
USER $USERNAME

# Entrypoint for the container
CMD ["/bin/zsh"]
