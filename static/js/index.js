(() => {
  const modal = document.getElementById('result-modal');
  const carousel = document.querySelector('[data-carousel]');

  const setupCarousel = () => {
    if (!carousel) return;

    const slides = Array.from(carousel.querySelectorAll('.carousel-slide'));
    const previousButton = carousel.querySelector('[data-carousel-prev]');
    const nextButton = carousel.querySelector('[data-carousel-next]');
    const status = carousel.querySelector('.carousel-status');
    if (!slides.length) return;

    let activeIndex = Math.max(0, slides.findIndex((slide) => slide.classList.contains('is-active')));

    const slideTitle = (slide) => slide.querySelector('figcaption')?.textContent.trim() || '';

    const syncSlides = (direction) => {
      carousel.classList.toggle('is-moving-backward', direction === 'backward');

      slides.forEach((slide, index) => {
        const isActive = index === activeIndex;
        const imageButton = slide.querySelector('.carousel-image-button');
        slide.classList.toggle('is-active', isActive);
        slide.setAttribute('aria-hidden', String(!isActive));
        if (imageButton) {
          if (isActive) {
            imageButton.removeAttribute('tabindex');
          } else {
            imageButton.setAttribute('tabindex', '-1');
          }
        }
      });

      if (status) {
        status.textContent = `Slide ${activeIndex + 1} of ${slides.length}: ${slideTitle(slides[activeIndex])}`;
      }
    };

    const moveTo = (nextIndex, direction) => {
      activeIndex = (nextIndex + slides.length) % slides.length;
      syncSlides(direction);
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

  setupCarousel();

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
