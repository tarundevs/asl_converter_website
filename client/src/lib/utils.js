import {clsx} from 'clsx.jsx'
import {twMerge} from 'tailwind-merge'

export const cn= (...inputs) => {
    return twMerge(clsx(inputs));
};