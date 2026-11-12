(() => {
  const modal = document.getElementById('result-modal');

  const setupCarousel = (carousel) => {
    if (!carousel) return;

    const slides = Array.from(carousel.querySelectorAll('.carousel-slide'));
    const previousButton = carousel.querySelector('[data-carousel-prev]');
    const nextButton = carousel.querySelector('[data-carousel-next]');
    const status = carousel.querySelector('.carousel-status');
    if (!slides.length) return;

    let activeIndex = Math.max(0, slides.findIndex((slide) => slide.classList.contains('is-active')));

    const dots = document.createElement('div');
    dots.className = 'carousel-dots';
    dots.setAttribute('aria-label', 'Slides');
    slides.forEach((_, index) => {
      const dot = document.createElement('button');
      dot.type = 'button';
      dot.className = 'carousel-dot';
      dot.setAttribute('aria-label', `Show slide ${index + 1}`);
      dot.addEventListener('click', () => {
        if (index === activeIndex) return;
        moveTo(index, index < activeIndex ? 'backward' : 'forward');
      });
      dots.append(dot);
    });
    carousel.append(dots);

    dots.addEventListener('wheel', (event) => {
      if (dots.scrollWidth <= dots.clientWidth) return;
      if (Math.abs(event.deltaY) <= Math.abs(event.deltaX)) return;
      event.preventDefault();
      dots.scrollLeft += event.deltaY;
    }, { passive: false });

    const slideTitle = (slide) => slide.querySelector('figcaption')?.textContent.trim() || '';

    const updateDots = () => {
      const dotButtons = dots.querySelectorAll('.carousel-dot');
      dotButtons.forEach((dot, index) => {
        const isActive = index === activeIndex;
        dot.classList.toggle('is-active', isActive);
        if (isActive) dot.setAttribute('aria-current', 'true');
        else dot.removeAttribute('aria-current');
      });
      const activeDot = dotButtons[activeIndex];
      if (!activeDot) return;
      const target = activeDot.offsetLeft - (dots.clientWidth - activeDot.offsetWidth) / 2;
      dots.scrollTo({ left: Math.max(0, target), behavior: 'smooth' });
    };

    const syncSlides = (direction) => {
      carousel.classList.toggle('is-moving-backward', direction === 'backward');

      slides.forEach((slide, index) => {
        const isActive = index === activeIndex;
        slide.classList.toggle('is-active', isActive);
        slide.setAttribute('aria-hidden', String(!isActive));
        slide.querySelectorAll('.carousel-image-button').forEach((imageButton) => {
          if (isActive) {
            imageButton.removeAttribute('tabindex');
          } else {
            imageButton.setAttribute('tabindex', '-1');
          }
        });
      });

      if (status) {
        const title = slideTitle(slides[activeIndex]);
        status.textContent = title
          ? `Slide ${activeIndex + 1} of ${slides.length}: ${title}`
          : `Slide ${activeIndex + 1} of ${slides.length}`;
      }

      const description = carousel.parentElement.querySelector('[data-live-description]');
      const descriptionText = slides[activeIndex].dataset.description;
      if (description && descriptionText) description.textContent = descriptionText;
      updateDots();
    };

    let runningAnimations = [];
    const moveTo = (nextIndex, direction) => {
      const targetIndex = (nextIndex + slides.length) % slides.length;
      if (targetIndex === activeIndex) return;
      runningAnimations.forEach((animation) => animation.cancel());
      runningAnimations = [];
      slides.forEach((slide) => slide.style.removeProperty('visibility'));
      const outgoing = slides[activeIndex];
      activeIndex = targetIndex;
      const incoming = slides[activeIndex];
      syncSlides(direction);
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

      const sign = direction === 'backward' ? -1 : 1;
      outgoing.style.visibility = 'visible';
      const options = { duration: 380, easing: 'cubic-bezier(.22,.61,.36,1)' };
      const leaving = outgoing.animate([
        { transform: 'translateX(0)' },
        { transform: `translateX(${-sign * 100}%)` }
      ], options);
      const entering = incoming.animate([
        { transform: `translateX(${sign * 100}%)` },
        { transform: 'translateX(0)' }
      ], options);
      runningAnimations = [leaving, entering];
      leaving.onfinish = () => { outgoing.style.removeProperty('visibility'); };
    };

    previousButton?.addEventListener('click', () => moveTo(activeIndex - 1, 'backward'));
    nextButton?.addEventListener('click', () => moveTo(activeIndex + 1, 'forward'));

    carousel.addEventListener('keydown', (event) => {
      if (event.key === 'ArrowLeft') {
        event.preventDefault();
        moveTo(activeIndex - 1, 'backward');
      }

      if (event.key === 'ArrowRight') {
        event.preventDefault();
        moveTo(activeIndex + 1, 'forward');
      }
    });

    syncSlides('forward');
  };

  document.querySelectorAll('[data-carousel]').forEach(setupCarousel);

  if (!modal) return;

  const modalImage = modal.querySelector('img');
  const modalCaption = modal.querySelector('figcaption');
  const closeButton = modal.querySelector('.modal-close');
  const imageButtons = document.querySelectorAll('.carousel-image-button');

  const closeModal = () => {
    modal.hidden = true;
    modalImage.removeAttribute('src');
    modalImage.removeAttribute('alt');
    modalCaption.textContent = '';
    document.body.classList.remove('has-open-modal');
  };

  const openModal = (button) => {
    const image = button.querySelector('img');
    modalImage.src = button.dataset.modalImage;
    modalImage.alt = image ? image.alt : '';
    modalCaption.textContent = button.dataset.modalCaption || '';
    modal.hidden = false;
    document.body.classList.add('has-open-modal');
    closeButton.focus();
  };

  imageButtons.forEach((button) => {
    button.addEventListener('click', () => openModal(button));
  });

  closeButton.addEventListener('click', closeModal);
  modal.addEventListener('click', (event) => {
    if (event.target === modal) closeModal();
  });

  document.addEventListener('keydown', (event) => {
    if (!modal.hidden && event.key === 'Escape') closeModal();
  });
})();
