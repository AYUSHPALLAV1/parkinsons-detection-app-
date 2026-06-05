// =========================================
// Performance: Lazy Video Loading
// Videos only load + autoplay when scrolled into view.
// This is the #1 fix for the homepage lag — previously
// ALL 4+ videos loaded and decoded simultaneously on page open.
// =========================================
(function initLazyVideos() {
    const lazyVideos = document.querySelectorAll('video[data-lazy-video]');
    if (!lazyVideos.length) return;

    const videoObserver = new IntersectionObserver((entries, obs) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const video = entry.target;
                // Swap data-src → src on all <source> children
                const sources = video.querySelectorAll('source[data-src]');
                sources.forEach(source => {
                    source.src = source.dataset.src;
                    source.removeAttribute('data-src');
                });
                video.load();
                video.play().catch(() => {}); // suppress autoplay policy errors
                obs.unobserve(video);
            }
        });
    }, { rootMargin: '200px' }); // start loading slightly before visible

    lazyVideos.forEach(v => videoObserver.observe(v));
})();

// =========================================
// Section Visibility Animation
// =========================================
const observerOptions = {
    root: null,
    rootMargin: '0px',
    threshold: 0.2
};

const observer = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
        }
    });
}, observerOptions);

document.querySelectorAll('section').forEach(section => {
    observer.observe(section);
});

// =========================================
// Dedicated Observer for Premium Video Cards (Section 2)
// Triggers fly-in animation + balloon text on first view
// =========================================
const videoSectionObserver = new IntersectionObserver((entries, obs) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            // Unobserve to trigger strictly the FIRST time the user enters
            obs.unobserve(entry.target);
            
            // Trigger Cards (they have internal staggered css delays)
            const videoFrames = entry.target.querySelectorAll('.video-frame');
            videoFrames.forEach(frame => {
                frame.classList.add('animate-in');
            });

            // Trigger Text Balloon Effect after cards settle (approx 1.2s logic)
            setTimeout(() => {
                const textNodes = entry.target.querySelectorAll('.video-description');
                textNodes.forEach((node, index) => {
                    // Stagger the text balloons slightly
                    setTimeout(() => {
                        node.classList.add('animate-text');
                    }, index * 150); 
                });
            }, 1200); 
        }
    });
}, { threshold: 0.3 });

const featureSection = document.getElementById('features');
if (featureSection) {
    videoSectionObserver.observe(featureSection);
}

// =========================================
// Animated Text Roller Logic
// =========================================
document.addEventListener('DOMContentLoaded', () => {
    const rollerWrapper = document.getElementById('roller-wrapper');
    if (rollerWrapper) {
        const totalItems = 4;
        let currentIndex = 0;
        setInterval(() => {
            currentIndex = (currentIndex + 1) % totalItems;
            rollerWrapper.style.transform = `translateY(calc(-${currentIndex} * 100% / ${totalItems}))`;
        }, 2000);
    }
});
