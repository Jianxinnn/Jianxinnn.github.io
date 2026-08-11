import Image from "next/image";

type BlogPostImageProps = {
  alt?: string;
  className: string;
  height: number;
  priority?: boolean;
  src: string;
  width: number;
};

function isRemoteImage(src: string) {
  return /^https?:\/\//.test(src);
}

export function BlogPostImage({
  alt = "",
  className,
  height,
  priority = false,
  src,
  width
}: BlogPostImageProps) {
  if (isRemoteImage(src)) {
    return (
      <img
        alt={alt}
        className={className}
        decoding="async"
        height={height}
        loading={priority ? "eager" : "lazy"}
        src={src}
        width={width}
      />
    );
  }

  return (
    <Image
      alt={alt}
      className={className}
      height={height}
      priority={priority}
      sizes="(max-width: 720px) 100vw, 320px"
      src={src}
      width={width}
    />
  );
}
