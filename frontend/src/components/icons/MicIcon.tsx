import React from 'react';
const MicIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="24" height="24" {...props}>
    <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.91-3c-.49 0-.9.39-.98.88C16.58 14.86 14.48 17 12 17s-4.58-2.14-4.93-5.12c-.08-.49-.49-.88-.98-.88-.55 0-1 .45-1 1 0 3.03 2.47 5.5 5.5 5.91V21h-2c-.55 0-1 .45-1 1s.45 1 1 1h6c.55 0 1-.45 1-1s-.45-1-1-1h-2v-2.09c3.03-.41 5.5-2.88 5.5-5.91 0-.55-.45-1-1-1z"/>
  </svg>
);
export default MicIcon;