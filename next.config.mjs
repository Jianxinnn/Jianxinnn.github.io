/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  trailingSlash: true,
  images: {
    unoptimized: true,
    localPatterns: [
      {
        pathname: "/images/**"
      }
    ]
  }
};

export default nextConfig;
